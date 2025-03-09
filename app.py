import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from langchain import hub
prompt = hub.pull("rlm/rag-prompt")


# Load and process documents
def load_and_process_data(file_path):
    split_tup = os.path.splitext(file_path)
    file_extension = split_tup[1].lower()
    if file_extension == ".txt":
        loader = TextLoader(file_path)
    elif file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise Exception("Invalid Document")
    
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks

# Initialize ChromaDB
def initialize_vector_db(chunks):
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=256)
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
    )
    vector_db = vector_store.from_documents(chunks, embeddings)
    return vector_db

# Set up QA chain
def create_qa_chain(vector_db):
    retriever = vector_db.as_retriever()
    llm = OpenAI()
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    return qa_chain

def generate(question, documents):
    docs_content = "\n\n".join(doc.page_content for doc in documents)
    messages = prompt.invoke({"question": question, "context": docs_content})
    llm = OpenAI()
    response = llm.invoke(messages)
    # print(response)
    return response

# Create an empty container
placeholder = st.empty()

actual_email = "demo@demo.com"
actual_password = "Test@1234"

# Insert a form in the container
with placeholder.form("login"):
    st.markdown("#### Enter your credentials")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    submit = st.form_submit_button("Login")
    
    
if st.session_state.is_authenticated == True or (submit and email == actual_email and password == actual_password):
    # If the form is submitted and the email and password are correct,
    # clear the form/container and display a success message
    placeholder.empty()
    st.session_state.is_authenticated = True
    # Streamlit UI
    st.title("RAG-based Chatbot")
    
    uploaded_file = st.file_uploader("Upload a PDF, DOCX and TXT file", type=["pdf", "docx", "txt"])
    
    if uploaded_file is not None:
        file_path = f"./{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        chunks = load_and_process_data(file_path)
        vector_db = initialize_vector_db(chunks)
        # qa_chain = create_qa_chain(vector_db)
        
        user_query = st.text_input("Ask a question about the document:")
        if user_query:
            results = vector_db.similarity_search(user_query)
    
            # print(results[0])
            response = generate(user_query, results)
            st.write("Answer:", response)

elif submit and email != actual_email and password != actual_password:
    st.error("Login failed")
else:
    pass
