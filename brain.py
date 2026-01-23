import os
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Initialize the ultra-fast Groq LLM
llm = ChatGroq(
    temperature=0.2, 
    model_name="llama3-70b-8192", 
    groq_api_key=os.getenv("GROQ_API_KEY")
)

class AvianBrain:
    def __init__(self):
        # Define Agents
        self.diagnostician = Agent(
            role='Avian Pathologist',
            goal='Analyze bird symptoms and identify potential health risks.',
            backstory='''You are a world-renowned avian vet. You are expert at spotting 
            subtle signs of illness in parrots, conures, and small birds. You are 
            empathetic but highly clinical.''',
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        self.care_specialist = Agent(
            role='Avian Care Specialist',
            goal='Provide immediate first-aid steps and long-term recovery advice.',
            backstory='''You specialize in avian rehabilitation and nutrition. 
            You translate complex medical diagnosis into easy-to-follow steps for bird owners.''',
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

    def process_request(self, bird_info):
        # Task 1: Analysis
        task1 = Task(
            description=f"Analyze the following bird health concerns: {bird_info}. Identify urgency level.",
            agent=self.diagnostician,
            expected_output="A summary of potential issues and an urgency rating (Low/Medium/High)."
        )

        # Task 2: Action Plan
        task2 = Task(
            description="Based on the pathologist's findings, provide 3 immediate first-aid steps.",
            agent=self.care_specialist,
            expected_output="A bulleted list of 3 actionable steps and 1 dietary recommendation."
        )

        # Create Crew
        crew = Crew(
            agents=[self.diagnostician, self.care_specialist],
            tasks=[task1, task2],
            process=Process.sequential
        )

        return crew.kickoff()
