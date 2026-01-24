import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from crewai import Agent, Task, Crew, Process

load_dotenv()

def load_rag_chain(query):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists("vector_db"):
        db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(query, k=2)
        return "\n".join([d.page_content for d in docs])
    return "Clinical Knowledge Base: Birds require stable temperatures and specialized diets."

def crew_ai_response(query, breed):
    # Switched to OpenAI GPT-4o for superior reasoning
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

    pathologist = Agent(
        role="Senior Avian Pathologist",
        goal=f"Identify clinical health risks for this {breed}",
        backstory="A world-leading vet with 25 years of experience in psittacine health.",
        llm=llm,
        verbose=True
    )

    behaviorist = Agent(
        role="Avian Ethologist",
        goal="Interpret behavioral stress and emotional triggers",
        backstory="Specializes in the psychological welfare of companion birds.",
        llm=llm,
        verbose=True
    )

    task = Task(
        description=f"Analyze {breed} concern: {query}. Cross-check symptoms with clinical data.",
        expected_output="A high-level professional diagnostic summary and recovery plan.",
        agent=pathologist
    )

    crew = Crew(agents=[pathologist, behaviorist], tasks=[task], process=Process.sequential)
    return str(crew.kickoff())
