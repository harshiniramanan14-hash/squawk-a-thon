import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from crewai import Agent, Task, Crew

load_dotenv()

DATA_PATH = "data/"
VECTOR_DB_PATH = "vector_db"

def build_vector_db():
    documents = []

    for folder in ["clinical_docs", "behavioural_docs"]:
        path = os.path.join(DATA_PATH, folder)
        for file in os.listdir(path):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(path, file))
                documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DB_PATH)

    return db

def load_rag_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever()
    )

def crew_ai_response(query, breed):
    vet_agent = Agent(
        role="Avian Veterinarian AI",
        goal="Diagnose and explain avian health & behavioral issues",
        backstory="Expert in bird medicine, ethology, and welfare",
        verbose=True
    )

    behavior_agent = Agent(
        role="Avian Behavior Specialist",
        goal="Analyze bird behavior and emotional states",
        backstory="Studies parrot psychology and environmental stress",
        verbose=True
    )

    task = Task(
        description=f"""
        Bird Breed: {breed}
        User Concern: {query}

        Provide:
        - Possible causes
        - Health risks
        - Behavioral insights
        - Immediate care tips
        - When to see a vet
        """,
        expected_output="Clear avian healthcare guidance",
        agents=[vet_agent, behavior_agent]
    )

    crew = Crew(
        agents=[vet_agent, behavior_agent],
        tasks=[task]
    )

    return crew.kickoff()
