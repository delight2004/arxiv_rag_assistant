from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# load the same embedding model we used for ingestion
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

#define the path where the database was saved
downloads_dir = Path("../data/downloaded_papers")
db_path = downloads_dir / "faiss_index"

#load the vector database from disk
db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

# define a simple query
query = "What is a Transformer model?"

# perform a similarity search to find the most relevant chunks
docs = db.similarity_search(query, k=4)

print("\n--- Retrieval Results ----")
print(f"Query: {query}")
print(f"Found {len(docs)} relevant chunks:")
for i, doc in enumerate(docs):
    print(f"\nChunk {i+1}")
    print(doc.page_content)
print("-" * 50)


# Create a prompt template
template = """
You are a helpful assistant. Answer the question based only on the following context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Instantiate the local LLM
ollama_llm = OllamaLLM(model="llama2")

# create a simple RAG chain
rag_chain = (
    {"context": db.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | ollama_llm
    | StrOutputParser()
)

#invoke the chain with your query
response = rag_chain.invoke(query)

print("\n--- Generated Answer ---")
print(response)
print("-" * 50)