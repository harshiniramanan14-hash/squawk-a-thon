import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Fixed Import path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS # Fixed Import path
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process

load_dotenv()

# Securely load keys to prevent TypeError
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def load_rag_chain(query):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Search local clinical docs
    if os.path.exists("vector_db"):
        db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(query, k=2)
        return "\n".join([d.page_content for d in docs])
    return "No clinical documents found. Using general avian knowledge base."

def crew_ai_response(query, breed):
    # Using Groq Llama 3 for speed and to fix "Native Provider" errors
    llm = ChatGroq(temperature=0.3, model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)

    pathologist = Agent(
        role="Avian Pathologist",
        goal=f"Analyze health markers for a {breed}",
        backstory="Expert in clinical avian medicine and diagnostic markers.",
        llm=llm
    )

    behaviorist = Agent(
        role="Avian Behavior Specialist",
        goal="Interpret stress signals and social behaviors",
        backstory="Specialist in parrot psychology and environmental health.",
        llm=llm
    )

    task = Task(
        description=f"Analyze this {breed} with symptoms: {query}. Provide causes and care steps.",
        expected_output="A structured specialist report.",
        agent=pathologist
    )

    crew = Crew(agents=[pathologist, behaviorist], tasks=[task], process=Process.sequential)
    return str(crew.kickoff())
