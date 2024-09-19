# Retrieval-Augmented Generation Application

This project is designed to work with the OpenAI API for text generation, utilizing documents in PDF format for retrieval-augmented generation (RAG). The app loads, processes, and indexes PDFs from the `data` folder, allowing users to query the document contents. The OpenAI model is used to generate responses based on document context.

## Features

- Loads and processes PDF documents from the `data` folder.
- Splits documents into manageable text chunks using the `RecursiveCharacterTextSplitter`.
- Stores embeddings using the Chroma vector store.
- Performs similarity searches to find the most relevant documents for a given query.
- Uses OpenAI API for text generation based on the retrieved document context.
- Easily modifiable to work with open-source models like HuggingFace models.

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python libraries (listed in `requirements.txt`)
