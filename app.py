from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

import streamlit as st

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Page Setup ---
st.set_page_config(page_title="Nirvana RAG Chatbot 💬", layout="centered")
st.title("🎤 Nirvana RAG Chatbot")

@st.cache_resource
def setup_qa_chain():
    try:
        loader = TextLoader("Data/Nirvana.txt", encoding='utf-8')
        documents = loader.load()
    except FileNotFoundError:
        st.error("Error: Data/Nirvana.txt not found. Please ensure the file exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading Data/Nirvana.txt: {e}")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) 
    docs = text_splitter.split_documents(documents)

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"Error initializing embeddings or vector store: {e}")
        st.stop()

    try:
        llm = OllamaLLM(model="mistral", temperature=0)
    except Exception as e:
        st.error(f"Error initializing Ollama LLM. Is Ollama running? Is 'mistral' pulled? Error: {e}")
        st.stop()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), 
        memory=memory, 
        verbose=False, 
        return_source_documents=True
    )
    return qa

# --- Constants ---
ASSISTANT_WELCOME_MESSAGE = "Hi! Ask me anything about Nirvana based on my knowledge base."

# --- Sidebar ---
with st.sidebar:
    st.header("Options")
    if st.button("Clear Chat History"):
        st.session_state.clear()
        st.session_state.messages = [{"role": "assistant", "content": ASSISTANT_WELCOME_MESSAGE}]
        st.rerun() 

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": ASSISTANT_WELCOME_MESSAGE}]

# --- Load QA Chain ---
try:
    qa_chain = setup_qa_chain()
except Exception as e:
    pass 

# --- Display Prior Chat Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# --- Chat Input ---
query = st.chat_input("E.g., Who is the lead singer?")

if query:
    # 1. Add User Message to State and Display
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Process Query and Get Response
    with st.spinner("Thinking..."):
        try:
            chat_history_for_chain = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    chat_history_for_chain.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history_for_chain.append(AIMessage(content=msg["content"]))

            # Invoke the chain, passing the query and the formatted chat history
            result = qa_chain.invoke({
                "question": query,
                "chat_history": chat_history_for_chain
            })

            # Extract answer and source documents
            response = result.get("answer", "Sorry, I faced an issue retrieving an answer.")
            source_docs = result.get("source_documents", [])

            # Prepare the message to store, including sources
            bot_message = {
                "role": "assistant",
                "content": response,
                "sources": source_docs 
            }

        except Exception as e:
            st.error(f"An error occurred while processing your query: {e}")
            bot_message = {
                "role": "assistant",
                "content": f"Sorry, an error occurred: {e}",
                "sources": []
            }

    # 3. Add Bot Response to State and Display
    st.session_state.messages.append(bot_message)
    with st.chat_message("assistant"):
        st.markdown(bot_message["content"])
        
        # --- Display Sources ---
        if bot_message["sources"]:
            with st.expander("Sources"):
                for i, doc in enumerate(bot_message["sources"]):
                    st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                    st.code(doc.page_content[:500] + "...")