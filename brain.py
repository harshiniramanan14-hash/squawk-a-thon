import os
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Initialize the ultra-fast Groq LLM (from your friend's code)
llm = ChatGroq(
    temperature=0.2, 
    model_name="llama3-70b-8192", 
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def load_rag_chain(query):
    """Your keyword-based RAG function"""
    knowledge_base = {
        "feather": "Feather issues: Often stress or nutrition. Provide enrichment.",
        "pluck": "Feather plucking: Increase toys and interaction. Check for boredom.",
        "sneeze": "Sneezing: Could be dust or allergy. Check air quality.",
        "letharg": "Lethargy: Serious sign. Ensure warmth (85°F), monitor closely.",
        "breath": "Breathing: Emergency if labored. Keep bird calm and contact vet.",
    }
    for keyword, info in knowledge_base.items():
        if keyword in query.lower():
            return info
    return "General avian care: Maintain stable temp and fresh water."

def crew_ai_response(query, breed):
    """Integrated Multi-Agent Analysis"""
    try:
        # Agent 1: Clinical Pathologist
        diagnostician = Agent(
            role='Avian Pathologist',
            goal=f'Analyze health concerns for this {breed}.',
            backstory='Expert at spotting subtle signs of illness in parrots.',
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        # Agent 2: Care Specialist
        care_specialist = Agent(
            role='Avian Care Specialist',
            goal='Provide immediate first-aid steps and long-term advice.',
            backstory='Specializes in avian rehabilitation and nutrition.',
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        # Task 1: Clinical Analysis
        task1 = Task(
            description=f"Analyze concern: {query} for a {breed}. Identify urgency.",
            agent=diagnostician,
            expected_output="Summary of potential issues and urgency rating (Low/Medium/High)."
        )

        # Task 2: Action Plan
        task2 = Task(
            description="Provide 3 immediate first-aid steps based on the analysis.",
            agent=care_specialist,
            expected_output="Bulleted list of 3 actionable steps and 1 dietary recommendation."
        )

        crew = Crew(
            agents=[diagnostician, care_specialist],
            tasks=[task1, task2],
            process=Process.sequential
        )

        result = crew.kickoff()
        return str(result)
        
    except Exception as e:
        return f"CrewAI Error: {str(e)}"
