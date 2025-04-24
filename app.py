from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

@st.cache_resource
def setup_qa_chain():
    # Load the text file and split it into chunks
    loader = TextLoader("Data/Nirvana.txt")
    documents = loader.load()

    # Split the documents into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Create a vector store using FAISS and HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # LLM and memory setup
    llm = OllamaLLM(model="gemma3:1b", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        verbose=True)
    
    return qa

st.set_page_config(page_title="RAG Chatbot 💬", layout="centered")
st.title("🎤 Nirvana RAG Chatbot")

qa_chain = setup_qa_chain()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask something about Nirvana:", placeholder="E.g. Who is the lead singer?")

if query:
    with st.spinner("Searching..."):
        answer = qa_chain.invoke(query)
        if isinstance(answer, dict) and "result" in answer:
            response = answer["result"]
        elif isinstance(answer, str):
            response = answer
        else:
            response = "Sorry, I couldn't retrieve a clear answer." # Handle unexpected output

        st.session_state.chat_history.append({"user": query, "bot": response})
        st.markdown(f"**Answer:** {response}")

for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")

