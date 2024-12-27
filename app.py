import fitz  # PyMuPDF
import re
from spellchecker import SpellChecker
spell = SpellChecker()
from langchain.schema import Document
from langchain.vectorstores import Chroma

from langchain.schema import HumanMessage
from langchain_community.chat_models.ollama import ChatOllama

# Step 1: Initialize the ChatOllama Model
local_model = "llama3:8b"
llm = ChatOllama(model=local_model)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# Step 2: Initialize the Vector Database with Chroma
persist_directory = "./chromadb_store"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use SentenceTransformers for embeddings
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

def extract_text_with_pymupdf(pdf_path):
    """
    Extract text from a PDF using PyMuPDF.
    """
    try:
        doc = fitz.open(pdf_path)
        page_text_dict = {}
        for i, page in enumerate(doc):
            page_text_dict[f"Page {i+1}"] = page.get_text() or "No text found on this page."
        return page_text_dict
    except Exception as e:
        return f"An error occurred while extracting text: {e}"

def correct_splits_and_typos(text):
    """
    Corrects split words and common typos using a spell checker.

    Args:
    - text (str): The text to be corrected.

    Returns:
    - str: Corrected text.
    """
    # Fix split words
    def merge_words(match):
        word1, word2 = match.groups()
        combined = word1 + word2
        if combined.lower() in spell:  # Check if the combined word is valid
            return combined
        return f"{word1} {word2}"  # Keep as is if not valid

    text = re.sub(r'\b(\w{2,})\s+(\w{1,4})\b', merge_words, text)

    # Spell-check and correct typos
    corrected_words = []
    for word in text.split():
        corrected_words.append(spell.correction(word) or word)
    return ' '.join(corrected_words)

def clean_and_correct_text_with_spellcheck(page_text_dict):
    """
    Cleans and corrects the extracted text by removing extra spaces, newlines, 
    and dynamically fixing split words and typos.

    Args:
    - page_text_dict (dict): Dictionary with page numbers as keys and text as values.

    Returns:
    - dict: Cleaned and corrected text for each page.
    """
    cleaned_text_dict = {}
    for page_num, text in page_text_dict.items():
        if text:  # Check if the page has text
            # Remove extra spaces and newlines
            text = re.sub(r'\s+', ' ', text.strip())
            # Dynamically correct split words and typos
            text = correct_splits_and_typos(text)
            cleaned_text_dict[page_num] = text
    return cleaned_text_dict

def chunk_text_by_phrase(clean_text_dict, chunk_size=300):
    """
    Chunks cleaned text by phrases, keeping track of the page numbers.

    Args:
    - clean_text_dict (dict): Dictionary with page numbers as keys and cleaned text as values.
    - chunk_size (int): Approximate size of each chunk in characters.

    Returns:
    - list of dict: List of dictionaries, where each dictionary represents a chunk with text and page number.
    """
    chunks = []
    for page_num, text in clean_text_dict.items():
        # Split into phrases by punctuation
        # Keep punctuation as part of the phrase
        phrases = re.split(r'([.!?])', text)

        chunk = ""
        for phrase in phrases:
            if len(chunk) + len(phrase) <= chunk_size:
                chunk += phrase
            else:
                if chunk.strip():
                    chunks.append({"page": page_num, "text": chunk.strip()})
                chunk = phrase
        if chunk.strip():
            chunks.append({"page": page_num, "text": chunk.strip()})
    return chunks

def add_chunks_to_chromadb(chunks):
    """
    Adds text chunks with metadata (page numbers) to ChromaDB.

    Args:
    - chunks (list of dict): List of chunks with 'page' and 'text' keys.
    """
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={"page": chunk["page"]}
        ) for chunk in chunks
    ]
    vectorstore.add_documents(documents)

class DocumentManager:
    def __init__(self):
        self.file_contexts = {}
        self.active_file = None

    def upload_book(self, file):
        """
        Uploads the book (PDF) and processes it for chunking and querying.
        """
        if file is None:
            return "No file uploaded!"

        pdf_path = file.name
        page_text_dict = extract_text_with_pymupdf(pdf_path)
        if isinstance(page_text_dict, str):  # Check if error occurred
            return page_text_dict

        cleaned_text_dict = clean_and_correct_text_with_spellcheck(page_text_dict)
        chunks = chunk_text_by_phrase(cleaned_text_dict, chunk_size=300)
        add_chunks_to_chromadb(chunks)

        # Store context
        self.file_contexts[file.name] = {
            "page_text_dict": page_text_dict,
            "cleaned_text_dict": cleaned_text_dict,
            "chunks": chunks,
        }
        self.active_file = file.name
        return f"Book '{file.name}' uploaded and processed successfully!"

    def set_active_file(self, file_name):
        """
        Sets the active file for querying.
        """
        if file_name in self.file_contexts:
            self.active_file = file_name
            return f"File '{file_name}' is now active for queries."
        else:
            return f"File '{file_name}' is not uploaded yet."

    def chat_with_book(self, question):
        """
        Handles user input and returns the answer and references as separate outputs.
        """
        if not self.active_file or self.active_file not in self.file_contexts:
            return "No active book set. Please upload or activate a book first.", "No references available."

        context = self.file_contexts[self.active_file]
        chunks = context["chunks"]
        if not chunks:
            return "The active book has no content to query.", "No references available."

        response = ask_book_question_with_references(question)

        # Split the response
        sections = response.split("### For Better Understanding, Refer to:")
        answer = sections[0].strip("### Answer:\n").strip()
        references = sections[1].strip() if len(sections) > 1 else "No references found."

        return answer, references
