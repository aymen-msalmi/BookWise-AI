
# BookWise AI 📚🤖

An AI-powered application that allows users to upload PDF books, extract their content, and ask natural language questions interactively. Get precise answers along with references from the book!

---

## Features

- **Upload PDFs**: Upload books or documents in PDF format.
- **Content Extraction**: Efficiently extract text using PyMuPDF.
- **Interactive Q&A**: Ask questions about the uploaded content and get AI-generated answers with references.
- **Contextual Answers**: Answers are generated based on the most relevant sections of the book.
- **References**: Provides references to the exact pages and text for further reading.

---

## Demo

![Demo Screenshot](img/demo-screenshot.png)  
_Upload a book, ask questions, and get answers!_

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/aymen-msalmi/BookWise-AI.git
   cd BookWise-AI
   ```

2. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**:
   Open the Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. **Access the application**:
   After running the final cell, the Gradio interface will be displayed.
   Open the provided link (e.g., `http://127.0.0.1:7860`) in your browser.

---

## Requirements

The application requires the following Python libraries:

- `gradio`: For creating the interactive web interface.
- `PyMuPDF` (`fitz`): For PDF text extraction.
- `langchain`: For managing document chunks and building queries.
- `chromadb`: For vectorizing and storing text representations.
- `sentence-transformers`: For generating embeddings using HuggingFace models.
- `autocorrect`: For text cleaning and spell-checking.

---
