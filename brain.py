import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from crewai import Agent, Task, Crew, Process

load_dotenv()

def load_rag_chain(query):
    # Fixes ImportError by using current LangChain paths
    HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists("vector_db"):
        db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(query, k=2)
        return "\n".join([d.page_content for d in docs])
    return "Base Knowledge: Avian health requires stable temperatures and species-specific nutrition."

def crew_ai_response(query, breed):
    # Fixes 401 Error: Ensure your .env has a valid sk-proj- key
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

    pathologist = Agent(
        role="Senior Avian Pathologist",
        goal=f"Analyze clinical health risks for this {breed}",
        backstory="World-renowned vet specializing in parrot diagnostics and pathology.",
        llm=llm,
        verbose=True
    )

    behaviorist = Agent(
        role="Avian Ethologist",
        goal="Interpret psychological stress and environmental triggers",
        backstory="Expert in bird psychology, social bonding, and stress-related behaviors.",
        llm=llm,
        verbose=True
    )

    task = Task(
        description=f"Breed: {breed}. Concern: {query}. Cross-reference symptoms with clinical data.",
        expected_output="A professional diagnostic report with clear recovery steps.",
        agent=pathologist
    )

    crew = Crew(agents=[pathologist, behaviorist], tasks=[task], process=Process.sequential)
    return str(crew.kickoff())
