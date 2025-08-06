



import streamlit as st

st.markdown("""
<style>
    .stTextInput > div > div > input {
        font-size: 16px;
        padding: 0.75em;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 0.75em 1.5em;
        font-size: 16px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

from rag_chain import generate_answer  # Your existing function

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📚 RAG-Powered Chatbot")
st.markdown("Ask questions based on the uploaded PDF documents")

query = st.text_input("Ask your question:", placeholder="e.g. What is the document about?")

if query:
    with st.spinner("Thinking..."):
        answer = generate_answer(query)
        st.success(answer)
