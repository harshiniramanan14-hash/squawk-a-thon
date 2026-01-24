import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from crewai import Agent, Task, Crew

load_dotenv()

DATA_PATH = "data"
VECTOR_DB_PATH = "vector_db"


# ---------- BUILD / LOAD VECTOR DB ----------

def build_or_load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    documents = []

    for folder in ["clinical_docs", "behavioural_docs"]:
        folder_path = os.path.join(DATA_PATH, folder)
        if not os.path.exists(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(folder_path, file))
                documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DB_PATH)

    return db


# ---------- RAG CONTEXT RETRIEVAL ----------

def get_rag_context(query: str) -> str:
    db = build_or_load_vector_db()
    retriever = db.as_retriever(search_kwargs={"k": 4})

    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(doc.page_content for doc in docs)


# ---------- CREW AI RESPONSE ----------

def crew_ai_response(query: str, breed: str) -> str:
    vet_agent = Agent(
        role="Avian Veterinarian",
        goal="Diagnose avian health problems",
        backstory="Expert in bird medicine and pathology",
        verbose=False
    )

    behavior_agent = Agent(
        role="Avian Behavior Specialist",
        goal="Analyze bird behavior and stress patterns",
        backstory="Expert in parrot psychology and welfare",
        verbose=False
    )

    task = Task(
        description=f"""
Bird Breed: {breed}

User Concern:
{query}

Give:
- Possible causes
- Health risks
- Behavioral interpretation
- Immediate care tips
- When to consult a vet
""",
        expected_output="Clear, structured avian healthcare guidance"
    )

    crew = Crew(
        agents=[vet_agent, behavior_agent],
        tasks=[task]
    )

    return crew.kickoff()
