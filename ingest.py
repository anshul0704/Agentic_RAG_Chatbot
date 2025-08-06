from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os


DATA_DIR= "data/"
CHROMA_DIR="embeddings/chroma/"
MODEL_NAME="all-MiniLM-L6-v2"

def load_and_chunk_docs():
    all_chunks=[]
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            loader=PyPDFLoader(os.path.join(DATA_DIR, filename))
            docs= loader.load()

            splitter= RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks=splitter.split_documents(docs)
            all_chunks.extend(chunks)
    return all_chunks

def embed_and_store(docs):
    embedding= HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db=Chroma.from_documents(docs, embedding, persist_directory=CHROMA_DIR)
    db.persist()

  
if __name__== "__main__":
    chunks=load_and_chunk_docs()
    embed_and_store(chunks)
    print("ChromaDB vectore created and saved")


