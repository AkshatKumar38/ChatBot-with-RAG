import os
import time
import warnings
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate

warnings.filterwarnings("ignore", category=DeprecationWarning)

# UI Setup
st.set_page_config(page_title="Nirvana RAG Chatbot 💬", layout="centered")
st.title("🎤 Nirvana RAG Chatbot")

# Constants
VECTORSTORE_DIR = "faiss_index"
DATA_FILE = "Data/Nirvana.txt"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
ASSISTANT_WELCOME_MESSAGE = "Hi! Ask me anything about Nirvana based on my knowledge base."

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": ASSISTANT_WELCOME_MESSAGE}]
if "timing_logs" not in st.session_state:
    st.session_state.timing_logs = []

# Load QA Chain with Caching
@st.cache_resource
def setup_qa_chain():
    start_total = time.time()
    try:
        if os.path.exists(VECTORSTORE_DIR):
            start_vs = time.time()
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.timing_logs.append(f"🔄 HuggingFace Embedding loaded in {time.time() - start_vs:.2f}s")
            vectorstore = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
            st.session_state.timing_logs.append(f"🔄 Vectorstore loaded in {time.time() - start_vs:.2f}s")
        else:
            start_embed = time.time()
            loader = TextLoader("Data/Nirvana.txt", encoding='utf-8')
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) 
            docs = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            vectorstore.save_local(VECTORSTORE_DIR)
            st.session_state.timing_logs.append(f"📦 New index built in {time.time() - start_embed:.2f}s")

    except Exception as e:
        st.error(f"Error setting up vectorstore: {e}")
        st.stop()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    try:
        llm = OllamaLLM(model="mistral", temperature=0)
    except Exception as e:
        st.error(f"Error initializing Ollama LLM: {e}")
        st.stop()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever, 
        memory=memory, 
        verbose=False, 
        return_source_documents=True
    )
    st.session_state.timing_logs.append(f"✅ Total chain setup: {time.time() - start_total:.2f}s")
    return qa, retriever, llm

qa_chain, retriever, llm = setup_qa_chain()

# Display Past Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("E.g., Who is the lead singer?")
if query:
    # 1. User Input
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. LLM Response
    with st.spinner("Thinking..."):
        try:
            start_process = time.time()
            chat_history_for_chain = []
            for msg in st.session_state.messages[-6:-1]:
                if msg["role"] == "user":
                    chat_history_for_chain.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history_for_chain.append(AIMessage(content=msg["content"]))

            docs = retriever.get_relevant_documents(query)

            start_llm = time.time()
            prompt_template = PromptTemplate.from_template(
                "You are a helpful assistant answering questions about Nirvana.\n"
                "Use the following context:\n\n{context}\n\n"
                "Question: {question}"
            )
            context = "\n\n".join([doc.page_content for doc in docs])
            final_prompt = prompt_template.format(context=context, question=query)
            response = llm.invoke(final_prompt)
            st.session_state.timing_logs.append(f"🧠 LLM response in {time.time() - start_llm:.2f}s")

            bot_message = {
                "role": "assistant",
                "content": response,
                "sources": docs
            }

        except Exception as e:
            st.error(f"❌ Error while processing query: {e}")
            bot_message = {
                "role": "assistant",
                "content": f"Sorry, an error occurred: {e}",
                "sources": []
            }

    # 3. Show LLMs Response
    st.session_state.messages.append(bot_message)
    with st.chat_message("assistant"):
        st.markdown(bot_message["content"])
        if bot_message["sources"]:
            with st.expander("Sources"):
                for i, doc in enumerate(bot_message["sources"]):
                    st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                    st.code(doc.page_content[:500] + "...")

with st.sidebar:
    if st.session_state.timing_logs:
        st.subheader("⏱️ Timing Logs")
        for log in st.session_state.timing_logs:
            st.write(log)
            
with st.sidebar:
    st.header("Options")
    if st.button("Clear Chat History"):
        st.session_state.clear()
        st.session_state.messages = [{"role": "assistant", "content": ASSISTANT_WELCOME_MESSAGE}]
        st.session_state.timing_logs = []
        if 'qa_chain' in globals() and qa_chain:
            qa_chain.memory.clear()
        st.rerun()

    if st.button("Rebuild Index"):
        import shutil
        if os.path.exists(VECTORSTORE_DIR):
            shutil.rmtree(VECTORSTORE_DIR)
        st.success("Index deleted. Reloading will now rebuild it.")
        st.rerun()