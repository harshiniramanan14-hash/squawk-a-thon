import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Fixed Import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # Fixed Import
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process

load_dotenv()

# Setup Vector DB Paths
VECTOR_DB_PATH = "vector_db"

def load_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Check if DB exists, if not, create an empty or placeholder one
    if os.path.exists(VECTOR_DB_PATH):
        db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        # Placeholder for hackathon demo if no files are pre-loaded
        db = FAISS.from_texts(["Initial knowledge base: Birds need fresh water and seeds."], embeddings)
        
    # Use Groq for the main logic (much faster than T5 for a hackathon)
    llm = ChatGroq(temperature=0.2, model_name="llama3-70b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))

    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

def crew_ai_response(query, breed):
    llm = ChatGroq(temperature=0.3, model_name="llama3-70b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))

    vet_agent = Agent(
        role="Avian Veterinarian AI",
        goal="Diagnose avian health issues",
        backstory="Senior vet with 20 years experience in exotic birds.",
        llm=llm
    )

    behavior_agent = Agent(
        role="Avian Behavior Specialist",
        goal="Analyze behavioral psychology",
        backstory="Expert in parrot social dynamics and stress behaviors.",
        llm=llm
    )

    task = Task(
        description=f"Breed: {breed}. Concern: {query}. Provide causes, risks, and 3 care steps.",
        expected_output="Clear medical and behavioral guidance.",
        agent=vet_agent
    )

    crew = Crew(agents=[vet_agent, behavior_agent], tasks=[task], process=Process.sequential)
    return crew.kickoff()
