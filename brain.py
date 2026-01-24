import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from crewai import Agent, Task, Crew
from litellm import completion

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# -------------------------------
# RAG SETUP
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# -------------------------------
# CREW AI AGENTS
# -------------------------------
avian_vet = Agent(
    role="Avian Veterinarian",
    goal="Diagnose and suggest healthcare guidance for birds",
    backstory="Expert in avian diseases, nutrition, and symptoms.",
    verbose=True
)

avian_behaviorist = Agent(
    role="Avian Behavior Analyst",
    goal="Understand bird behavior and emotional state",
    backstory="Specialist in bird psychology and behavioral patterns.",
    verbose=True
)

# -------------------------------
# MAIN AI LOGIC
# -------------------------------
def squawk_ai(query, breed, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    context_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in context_docs])

    vet_task = Task(
        description=f"""
        Bird Breed: {breed}
        User Concern: {query}

        Context:
        {context}

        Give health analysis, causes, and care advice.
        """,
        agent=avian_vet
    )

    behavior_task = Task(
        description=f"""
        Bird Breed: {breed}
        User Concern: {query}

        Analyze emotional and behavioral patterns.
        """,
        agent=avian_behaviorist
    )

    crew = Crew(
        agents=[avian_vet, avian_behaviorist],
        tasks=[vet_task, behavior_task]
    )

    result = crew.kickoff()
    return result
