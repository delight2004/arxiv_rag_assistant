# ArXiv RAG Assistant

This project is a Retrieval-Augmented Generation (RAG) assistant that uses ArXiv papers as its knowledge base. It can download papers, process them, and answer questions based on their content.

## Features

-   **ArXiv Paper Downloader**: Downloads papers from ArXiv based on search queries or paper IDs.
-   **Text Processing**: Extracts text from PDFs using GROBID and splits it into manageable chunks.
-   **Vector Store**: Creates a FAISS vector store from the processed text for efficient retrieval.
-   **RAG Chain**: Uses a RAG chain with a local LLM (Ollama) to answer questions based on the retrieved context.

## Getting Started

### Prerequisites

-   Python 3.8+
-   [Ollama](https://ollama.ai/) installed and running
-   [GROBID](https://grobid.readthedocs.io/en/latest/Grobid-docker/) running in a Docker container

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/delight2004/arxiv_rag_assistant.git
    cd arxiv_rag_assistant
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the ingestion script to download and process the ArXiv papers:**

    ```bash
    python src/ingest.py
    ```

2.  **Run the application to start the RAG assistant:**

    ```bash
    python src/app.py
    ```

## Project Structure

```
.
├── data/
│   └── downloaded_papers/
│       ├── *.pdf
│       ├── *.xml
│       └── faiss_index/
├── src/
│   ├── app.py          # Main application file
│   ├── ingest.py       # Ingestion script for processing papers
│   └── main.py         # FastAPI application
├── .gitignore
├── README.md
└── requirements.txt
```

## Dependencies

-   `arxiv`
-   `beautifulsoup4`
-   `fastapi`
-   `faiss-cpu`
-   `grobid-client`
-f   `langchain`
-   `langchain-community`
-   `langchain-huggingface`
-   `langchain-text-splitters`
-   `ollama`
-   `sentence-transformers`
-   `uvicorn`

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
