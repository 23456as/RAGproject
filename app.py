import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
import os
import tempfile

st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="wide")

# SIMPLE CSS - no external resources
st.markdown("""
<style>
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
.stTextInput input { border-radius: 20px !important; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chain" not in st.session_state:
    st.session_state.chain = None

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get free: https://console.groq.com/keys")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("## 📁 Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and groq_api_key and st.button("📤 Process Files"):
        with st.spinner("Processing..."):
            try:
                all_data = []
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.getbuffer())
                        loader = UnstructuredPDFLoader(file_path=tmp.name)
                        all_data.extend(loader.load())
                        os.unlink(tmp.name)
                
                if all_data:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.split_documents(all_data)
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vector_db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
                    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
                    
                    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=groq_api_key)
                    
                    template = """Answer based on context:

Context:
{context}

Q: {question}
A:"""
                    
                    prompt = ChatPromptTemplate.from_template(template)
                    chain = (
                        {"context": retriever, "question": RunnablePassthrough()}
                        | prompt | llm | StrOutputParser()
                    )
                    
                    st.session_state.chain = chain
                    st.session_state.vector_db = vector_db
                    st.sidebar.success("✅ Ready!")
                    st.session_state.messages = []
                    
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")

st.title("💬 RAG Document Chat")

if st.session_state.chain:
    for message in st.session_state.messages:
        with st.chat_message("user" if message["role"] == "user" else "assistant"):
            st.write(message["content"])
    
    user_input = st.chat_input("Ask about your documents...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chain.invoke(user_input)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")

else:
    st.info("👈 Upload PDFs in the sidebar to get started")
    st.markdown("""
    ## Getting Started
    1. Get API Key: https://console.groq.com/keys
    2. Paste in sidebar
    3. Upload PDFs
    4. Click "Process Files"
    5. Start chatting!
    """)
