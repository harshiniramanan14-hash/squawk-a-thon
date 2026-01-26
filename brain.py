import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process

load_dotenv()

def load_rag_chain(query):
    """Simplified RAG - remove HuggingFace dependency"""
    try:
        # Check if vector database exists
        if os.path.exists("vector_db") and os.path.exists("vector_db/index.faiss"):
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                from langchain_community.vectorstores import FAISS
                
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                db = FAISS.load_local(
                    "vector_db", 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                docs = db.similarity_search(query, k=2)
                return "\n".join([d.page_content for d in docs])
            except:
                # If HuggingFace fails, use simple knowledge base
                pass
    except:
        pass
    
    # Fallback knowledge
    fallback_knowledge = {
        "feather plucking": "Common causes: stress, boredom, nutritional deficiency, parasites, allergies.",
        "lethargy": "Possible causes: infection, metabolic disorder, toxicity, heart disease.",
        "wing drooping": "Could indicate: injury, weakness, neurological issue, pain.",
        "sneezing": "Possible: respiratory infection, allergy, dust, foreign body.",
        "diarrhea": "Causes: bacterial infection, dietary change, parasites, liver disease."
    }
    
    # Find relevant knowledge
    for key in fallback_knowledge:
        if key in query.lower():
            return fallback_knowledge[key]
    
    return "Avian health requires: proper temperature (75-85°F), balanced nutrition, clean environment, regular vet checkups."

def crew_ai_response(query, breed):
    """Generate AI response using CrewAI with fallback"""
    try:
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_key or len(openai_key) < 20:
            return "OpenAI API key not configured properly."
        
        # Use simpler model
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # More reliable than gpt-4
            api_key=openai_key,
            temperature=0.7,
            max_tokens=800
        )
        
        # Simplified single agent
        avian_vet = Agent(
            role="Senior Avian Veterinarian",
            goal=f"Provide accurate health assessment for this {breed}",
            backstory="You are Dr. Feathers with 20+ years experience treating exotic birds.",
            llm=llm,
            verbose=False
        )
        
        task = Task(
            description=f"""
            BREED: {breed}
            SYMPTOMS: {query}
            
            Provide a veterinary assessment in this format:
            
            1. **Likely Causes** (list 3-5 possibilities)
            2. **Immediate Care** (next 24 hours)
            3. **Home Monitoring** (what to watch for)
            4. **Dietary Advice** (specific to {breed})
            5. **Vet Visit Indicators** (when to seek help)
            
            Be concise, practical, and avoid medical jargon.
            """,
            expected_output="Structured veterinary advice",
            agent=avian_vet
        )
        
        crew = Crew(
            agents=[avian_vet],
            tasks=[task],
            verbose=False
        )
        
        result = crew.kickoff()
        return str(result)
        
    except Exception as e:
        return f"Diagnostic system temporarily unavailable. General advice: Keep {breed} warm, hydrated, and in a quiet environment. Monitor closely."

if __name__ == "__main__":
    # Test the function
    test_query = "My parrot is plucking feathers"
    print(load_rag_chain(test_query))
