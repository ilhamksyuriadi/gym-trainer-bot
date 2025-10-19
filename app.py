import streamlit as st
import os
import sqlite3
import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

class GymTrainerBot:
    def __init__(self):
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store_path = "vector_store"
        self.db_path = "database/chat_history.db"
        
    def initialize_database(self):
        """Initialize SQLite database for chat history"""
        os.makedirs("database", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
            )
        ''')
        conn.commit()
        conn.close()
    
    def setup_rag_system(self):
        """Load PDFs, create vector store, and setup RAG chain"""
        # Check if vector store already exists
        if os.path.exists(self.vector_store_path) and len(os.listdir(self.vector_store_path)) > 0:
            st.info("Loading existing knowledge base...")
            vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings
            )
        else:
            st.info("Processing your gym documents...")
            # Load and process PDFs
            documents = []
            pdf_folder = "documents"
            
            for pdf_file in os.listdir(pdf_folder):
                if pdf_file.endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
                    documents.extend(loader.load())
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create vector store
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.vector_store_path
            )
            vector_store.persist()
            st.success(f"Processed {len(documents)} PDFs into {len(chunks)} chunks!")
        
        return vector_store
    
    def create_retrieval_chain(self, vector_store):
        """Create the RAG chain with gym-specific prompt"""
        # Gym trainer specific prompt
        prompt_template = """
        You are an expert gym personal trainer and fitness coach. You provide safe, effective, and personalized fitness advice.
        
        Use the following context from fitness documents to answer the question. If you don't know the answer based on the context, say so rather than making things up.
        
        Context: {context}
        
        Question: {input}
        
        Answer as a professional personal trainer:
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.3
        )
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
    
    def save_message(self, session_id, role, message):
        """Save message to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ensure session exists
        cursor.execute(
            "INSERT OR IGNORE INTO chat_sessions (session_id) VALUES (?)",
            (session_id,)
        )
        
        # Save message
        cursor.execute(
            "INSERT INTO chat_messages (session_id, role, message) VALUES (?, ?, ?)",
            (session_id, role, message)
        )
        
        conn.commit()
        conn.close()
    
    def get_chat_history(self, session_id):
        """Retrieve chat history for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT role, message, timestamp FROM chat_messages WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )
        
        messages = cursor.fetchall()
        conn.close()
        return messages

def main():
    st.set_page_config(
        page_title="Gym Personal Trainer Bot",
        page_icon="💪",
        layout="wide"
    )
    
    st.title("💪 Gym Personal Trainer Bot")
    st.markdown("Your AI fitness coach - get personalized workout and nutrition advice!")
    
    # Initialize bot
    if 'bot' not in st.session_state:
        st.session_state.bot = GymTrainerBot()
        st.session_state.bot.initialize_database()
    
    # Initialize session ID
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Sidebar for setup
    with st.sidebar:
        st.header("⚙️ Setup")
        if st.button("🔄 Initialize Knowledge Base"):
            with st.spinner("Setting up RAG system..."):
                st.session_state.vector_store = st.session_state.bot.setup_rag_system()
                st.session_state.retrieval_chain = st.session_state.bot.create_retrieval_chain(st.session_state.vector_store)
                st.success("Ready to train! 💪")
        
        st.markdown("---")
        st.header("📊 Session Info")
        st.write(f"Session: {st.session_state.session_id}")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your fitness question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.bot.save_message(st.session_state.session_id, "user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        if 'retrieval_chain' in st.session_state:
            with st.chat_message("assistant"):
                with st.spinner("Thinking about your fitness goals..."):
                    try:
                        response = st.session_state.retrieval_chain.invoke({"input": prompt})
                        answer = response["answer"]
                        
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.session_state.bot.save_message(st.session_state.session_id, "assistant", answer)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            with st.chat_message("assistant"):
                st.warning("Please initialize the knowledge base first using the button in the sidebar.")

if __name__ == "__main__":
    main()