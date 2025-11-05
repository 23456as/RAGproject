import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from PyPDF2 import PdfReader
import tempfile
import os

st.set_page_config(page_title="RAG Chat", page_icon="💬", layout="wide")

# ---- Custom CSS for ChatGPT Look ----
st.markdown('''
    <style>
    .main {background-color: #f7faff;}
    .stChatInputContainer {margin-bottom: 0;}
    div[data-testid="stChatMessage"] {max-width: 650px; margin-left:auto; margin-right:auto;}
    div.user-bubble {background: linear-gradient(135deg,#667eea 0%,#764ba2 100%!important); color: white!important; border-radius: 20px; padding:16px 20px;margin-bottom:10px; margin-top:14px;}
    div.assistant-bubble {background: #f2f2f6!important;color: #222!important; border-radius: 20px; padding:16px 20px;margin-bottom:10px; margin-top:14px;}
    .stChatInput input {border-radius: 32px!important; font-size: 19px!important; padding:18px 24px!important; border:1.5px solid #d1d5db;}
    .block-container {padding-top:24px;}
    </style>
''', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chain" not in st.session_state:
    st.session_state.chain = None

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", help="Get free: https://console.groq.com/keys")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown("## 📁 Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            st.caption(f"📄 {file.name} ({file.size / 1024:.1f} KB)")
    if uploaded_files and groq_api_key and st.button("📤 Process Files"):
        with st.spinner("Processing..."):
            try:
                all_data = []
                valid_files = 0
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.getbuffer())
                        tmp_path = tmp.name
                    try:
                        # Check if PDF is encrypted
                        with open(tmp_path, "rb") as f:
                            reader = PdfReader(f)
                            if reader.is_encrypted:
                                st.sidebar.warning(f"⚠️ {file.name} is encrypted, skipping")
                                os.unlink(tmp_path)
                                continue
                            if len(reader.pages) == 0:
                                st.sidebar.warning(f"⚠️ {file.name} has no pages, skipping")
                                os.unlink(tmp_path)
                                continue
                        loader = PDFPlumberLoader(file_path=tmp_path)
                        docs = loader.load()
                        if docs:
                            all_data.extend(docs)
                            valid_files += 1
                            st.sidebar.success(f"✓ {file.name} ({len(docs)} pages)")
                        else:
                            st.sidebar.warning(f"⚠️ {file.name} loaded but no text extracted")
                    except Exception as e:
                        st.sidebar.error(f"✗ {file.name}: {str(e)[:50]}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                if not all_data:
                    st.sidebar.error("No valid documents loaded!")
                    st.stop()
                st.sidebar.info(f"Processing {valid_files} file(s)...")
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(all_data)
                st.sidebar.success(f"✓ Created {len(chunks)} chunks")
                st.sidebar.info("Loading embeddings...")
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.sidebar.success("✓ Embeddings ready")
                st.sidebar.info("Creating vector database...")
                vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                st.sidebar.success("✓ Vector DB created")
                retriever = vector_db.as_retriever(search_kwargs={"k": 3})
                llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    groq_api_key=groq_api_key
                )
                template = '''You are a helpful assistant. Be concise and specific. Use only the context provided.

Context:
{context}

Question: {question}

Answer:'''
                prompt = ChatPromptTemplate.from_template(template)
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt | llm | StrOutputParser()
                )
                st.session_state.chain = chain
                st.session_state.vector_db = vector_db
                st.sidebar.success("✅ System Ready!")
                st.session_state.messages = []
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")

# ---- Main Chat Area ----
st.markdown("<h2 style='text-align:center;color:#764ba2;'>💬 RAG ChatGPT Document Q&A</h2>", unsafe_allow_html=True)
if st.session_state.chain:
    # Display chat history in bubbles
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div class='user-bubble'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-bubble'>{message['content']}</div>", unsafe_allow_html=True)
    user_input = st.chat_input("Type your question and hit Enter...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f"<div class='user-bubble'>{user_input}</div>", unsafe_allow_html=True)
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chain.invoke(user_input)
                st.markdown(f"<div class='assistant-bubble'>{response}</div>", unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
else:
    st.info("👈 Upload PDFs and enter your Groq API key to get started!")
    st.markdown('''
        <div style='text-align:center;'>
        <br><b>How to Use:</b><br><br>
        <ol style='max-width:440px;margin:0 auto;text-align:left;'>
            <li>Get API Key: <a href='https://console.groq.com/keys' target='_blank'>https://console.groq.com/keys</a></li>
            <li>Paste it in the sidebar</li>
            <li>Upload PDFs and click <i>Process Files</i></li>
            <li>Ask questions in the chat box!</li>
        </ol>
        <br>
        <b>Features:</b>
        <ul style='max-width:440px;margin:0 auto;text-align:left;'>
            <li>✅ ChatGPT-style interface (big chat box, rounded bubbles)</li>
            <li>✅ Supports scanned/text PDFs (skips encrypted)</li>
            <li>✅ Modern, centered layout</li>
        </ul>
        </div>
    ''', unsafe_allow_html=True)
