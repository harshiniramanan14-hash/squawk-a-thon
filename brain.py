import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from crewai import Agent, Task, Crew, Process

load_dotenv()

def load_rag_chain(query):
    """Load RAG context from vector database"""
    try:
        # FIXED: Proper embeddings initialization
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if os.path.exists("vector_db"):
            # FIXED: Added error handling for missing files
            db = FAISS.load_local(
                "vector_db", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            docs = db.similarity_search(query, k=2)
            return "\n".join([d.page_content for d in docs])
        else:
            st.warning("⚠️ Vector database not found. Using base knowledge.")
            return "Base Knowledge: Avian health requires stable temperatures and species-specific nutrition."
            
    except Exception as e:
        return f"RAG Error: {str(e)}. Using fallback knowledge."

def crew_ai_response(query, breed):
    """Generate AI response using CrewAI"""
    try:
        # FIXED: Validate API key before use
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or " " in openai_key:
            return "❌ ERROR: Invalid OpenAI API key. Please check your .env file."
        
        llm = ChatOpenAI(
            model="gpt-4o", 
            openai_api_key=openai_key,
            temperature=0.7
        )

        # Create agents
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

        # Create task
        task = Task(
            description=f"Breed: {breed}. Concern: {query}. Cross-reference symptoms with clinical data.",
            expected_output="A professional diagnostic report with clear recovery steps.",
            agent=pathologist
        )

        # Create and run crew
        crew = Crew(
            agents=[pathologist, behaviorist], 
            tasks=[task], 
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        return str(result)
        
    except Exception as e:
        return f"❌ CrewAI Error: {str(e)}"
