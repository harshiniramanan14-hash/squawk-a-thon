import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from crewai import Agent, Task, Crew, Process

load_dotenv()

def load_rag_chain(query):
    # Fixes ImportError by using correct LangChain paths
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists("vector_db"):
        db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(query, k=2)
        return "\n".join([d.page_content for d in docs])
    return "Clinical Knowledge: Avian health requires specific humidity and high-quality pellet diets."

def crew_ai_response(query, breed):
    # Fixes 401 Error by ensuring key is loaded correctly
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

    pathologist = Agent(
        role="Senior Avian Pathologist",
        goal=f"Analyze medical risks for this {breed}",
        backstory="A specialized avian vet focusing on diagnostic medicine.",
        llm=llm
    )

    behaviorist = Agent(
        role="Avian Ethologist",
        goal="Interpret behavioral stress and psychological triggers",
        backstory="Expert in psittacine behavior and social enrichment.",
        llm=llm
    )

    task = Task(
        description=f"Breed: {breed}. Concern: {query}. Provide a clinical/behavioral diagnosis.",
        expected_output="A professional healthcare summary with actionable care steps.",
        agent=pathologist
    )

    crew = Crew(agents=[pathologist, behaviorist], tasks=[task], process=Process.sequential)
    return str(crew.kickoff())
