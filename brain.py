import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process

load_dotenv()

def load_rag_chain(query):
    """Simplified RAG function"""
    try:
        # Simple knowledge base - you can expand this
        knowledge_base = {
            "feather": "Feather issues: Often stress, nutrition, or medical. Provide enrichment, review diet.",
            "pluck": "Feather plucking: Common in bored/stressed birds. Increase toys, interaction.",
            "sneeze": "Sneezing: Could be dust, infection, allergy. Check environment, consult vet if persistent.",
            "letharg": "Lethargy: Possible infection, metabolic issue. Ensure warmth, monitor closely.",
            "eat": "Appetite changes: Monitor weight. Offer favorite foods. Vet if not eating 24h.",
            "breath": "Breathing issues: Emergency if labored. Keep calm, warm, contact vet immediately.",
        }
        
        for keyword, info in knowledge_base.items():
            if keyword in query.lower():
                return info
        
        return "General avian care: Maintain 75-85°F, fresh water daily, balanced diet, reduce stress."
        
    except Exception as e:
        return f"Base knowledge: {str(e)[:100]}"

def crew_ai_response(query, breed):
    """Generate AI response using CrewAI"""
    try:
        # Get API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return "Error: OPENAI_API_KEY not found in environment variables"
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4",
            api_key=openai_key,
            temperature=0.7
        )
        
        # Create agents with specific roles
        vet_agent = Agent(
            role="Senior Avian Veterinarian",
            goal=f"Diagnose health issues for {breed} parrots",
            backstory="You are Dr. Feathers, board-certified avian vet with 30 years experience.",
            llm=llm,
            verbose=True
        )
        
        behavior_agent = Agent(
            role="Avian Behavior Specialist",
            goal="Analyze behavioral and psychological factors",
            backstory="You are Dr. Talon, avian ethologist specializing in parrot behavior.",
            llm=llm,
            verbose=True
        )
        
        # Create tasks
        diagnosis_task = Task(
            description=f"""
            Analyze this case for a {breed} parrot:
            
            Symptoms: {query}
            
            Provide a veterinary assessment including:
            1. Likely causes
            2. Immediate care needed
            3. When to see a vet
            4. Follow-up monitoring
            """,
            expected_output="Structured veterinary assessment",
            agent=vet_agent
        )
        
        behavior_task = Task(
            description=f"""
            For a {breed} with these symptoms: {query}
            
            Analyze:
            1. Behavioral red flags
            2. Environmental factors
            3. Stress indicators
            4. Enrichment recommendations
            """,
            expected_output="Behavioral analysis and recommendations",
            agent=behavior_agent
        )
        
        # Create and run crew
        crew = Crew(
            agents=[vet_agent, behavior_agent],
            tasks=[diagnosis_task, behavior_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        return str(result)
        
    except Exception as e:
        # Fallback response
        return f"""
        # 🩺 Avian Health Guidance for {breed}
        
        **For symptoms:** {query}
        
        ## Basic Care Protocol:
        1. **Monitor closely** - Check every 2-4 hours
        2. **Ensure basics** - Fresh water, proper temperature (75-85°F)
        3. **Reduce stress** - Quiet environment, familiar routine
        4. **Contact avian vet** if symptoms persist or worsen
        
        ## Emergency Signs (Seek Immediate Vet):
        - Difficulty breathing
        - Bleeding that doesn't stop
        - Inability to perch
        - Seizures or collapse
        
        *Note: AI diagnostic system experienced technical issue. {str(e)[:100]}*
        
        **Emergency Contact:** 1-800-AVIAN-VET
        """

# Test function
if __name__ == "__main__":
    # Test the system
    test_query = "My parrot is plucking feathers"
    test_breed = "African Grey"
    
    print("Testing RAG system...")
    context = load_rag_chain(test_query)
    print(f"RAG Context: {context[:100]}...")
    
    print("\nTesting CrewAI system...")
    response = crew_ai_response(test_query, test_breed)
    print(f"CrewAI Response: {response[:200]}...")
