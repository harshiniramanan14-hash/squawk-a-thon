import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub

from crewai import Agent, Task, Crew


load_dotenv()

# Setup Vector DB Paths
VECTOR_DB_PATH = "vector_db"

def load_rag_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )

    return retriever, llm


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
