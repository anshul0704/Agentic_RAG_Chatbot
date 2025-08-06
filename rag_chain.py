from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.tools import Tool

CHROMA_DIR = "embeddings/chroma/"
#MODEL_NAME = "all-Mini_L6-v2"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


embedding= HuggingFaceEmbeddings(model_name=MODEL_NAME)
db=Chroma(persist_directory= CHROMA_DIR, embedding_function= embedding)
retriever= db.as_retriever()
llm=Ollama(model= "mistral")
#llm = Ollama(model="llama3")

rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

def generate_answer(query):
    return rag_chain.run(query)


rag_tool = Tool(
    name="rag_chain",
    func=generate_answer,
    description="A tool that uses a RAG chain to answer questions based on a vector database of documents. "
)