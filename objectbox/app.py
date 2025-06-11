from langchain.chat_models import ChatOpenAI
import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_objectbox.vectorstores import ObjectBox
from dotenv import load_dotenv
import time

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.title("Objectbox VectorstoreDB with GPT-3.5 Demo")

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide the most accurate response to the question.

Context:
{context}

Question: {input}
"""
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader(r"C:\Users\hfwal\OneDrive\Desktop\portfolioproject\langchain series\.venv\objectbox\pdffile")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = ObjectBox.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings,
            embedding_dimensions=768
        )

input_prompt = st.text_input("Enter your question from documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("ObjectBox database is ready.")

if input_prompt:
    if "vectors" not in st.session_state:
        st.error("Please embed documents first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': input_prompt})
        st.write("Response time:", time.process_time() - start)
        st.write(response['answer'])

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response.get('context', [])):
                st.write(doc.page_content)
                st.write("--------------------------")
