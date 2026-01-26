import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from crewai import Agent, Task, Crew, Process

load_dotenv()

def load_rag_chain(query):
    """Load RAG context from vector database"""
    try:
        # FIXED: Proper embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if os.path.exists("vector_db") and os.path.exists("vector_db/index.faiss"):
            db = FAISS.load_local(
                "vector_db", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            docs = db.similarity_search(query, k=2)
            return "\n".join([d.page_content for d in docs])
        return "Base Knowledge: Avian health requires stable temperatures (75-85°F) and species-specific nutrition. Common issues: feather plucking (stress/nutrition), lethargy (infection/metabolic)."
            
    except Exception as e:
        return f"Base Knowledge (RAG Error): {str(e)}"

def crew_ai_response(query, breed):
    """Generate AI response using CrewAI"""
    try:
        # DEBUG: Check API key
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_key or " " in openai_key or len(openai_key) < 30:
            return "❌ ERROR: Invalid OpenAI API key format. Please regenerate at platform.openai.com/api-keys"
        
        # FIXED: Set explicit parameters
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # Changed to cheaper/more available model
            api_key=openai_key,
            temperature=0.7,
            max_tokens=1000
        )

        # SINGLE AGENT SIMPLIFICATION (Fix CrewAI issue)
        avian_specialist = Agent(
            role="Senior Avian Veterinarian Specialist",
            goal=f"Diagnose health issues for {breed} parrots and provide actionable advice",
            backstory="""You are Dr. Aviana Feathers, a board-certified avian veterinarian with 25 years 
            experience treating exotic birds. You specialize in psittacine medicine and have published 
            research on parrot nutrition and behavioral health.""",
            llm=llm,
            verbose=False,
            allow_delegation=False  # Important: Prevent delegation errors
        )

        # SINGLE TASK (Simplify Crew structure)
        diagnostic_task = Task(
            description=f"""
            BREED: {breed}
            CONCERN: {query}
            
            Please provide a comprehensive diagnostic assessment including:
            1. **Likely Causes** (ranked by probability)
            2. **Immediate Actions** (within 24 hours)
            3. **Dietary Recommendations**
            4. **When to See a Veterinarian** (red flags)
            5. **Preventive Measures**
            
            Format in clear, actionable bullet points.
            """,
            expected_output="A professional avian veterinary report with clear recommendations",
            agent=avian_specialist,
            context=[query]
        )

        # SIMPLIFIED CREW (Just one agent/task)
        crew = Crew(
            agents=[avian_specialist],
            tasks=[diagnostic_task],
            verbose=True,
            process=Process.sequential,
            memory=False  # Disable memory to simplify
        )
        
        result = crew.kickoff()
        return str(result)
        
    except Exception as e:
        return f"⚠️ Diagnostic System Temporarily Unavailable\n\nAs an avian health assistant, I can provide general guidance:\n\nFor {breed} with {query}:\n• Maintain temperature 75-85°F\n• Ensure fresh water daily\n• Monitor droppings and appetite\n• Reduce stress factors\n\nError Details: {str(e)[:200]}"
