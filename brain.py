import os
import sys
import json
import re
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import CrewAI and LangChain components
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain.agents import initialize_agent
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
class Config:
    """Configuration settings for the AI system"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN", "")
    
    # Model settings
    OPENAI_MODEL = "gpt-4o"  # or "gpt-4-turbo", "gpt-3.5-turbo"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Better than MiniLM
    
    # CrewAI settings
    MAX_ITERATIONS = 3
    VERBOSE = True
    
    # RAG settings
    SIMILARITY_SEARCH_K = 3
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Paths
    VECTOR_DB_PATH = "vector_db"
    KNOWLEDGE_BASE_PATH = "knowledge_base"
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return True

# --- KNOWLEDGE BASE MANAGEMENT ---
class AvianKnowledgeBase:
    """Manage avian medical knowledge"""
    
    # Core avian medical knowledge
    CORE_KNOWLEDGE = {
        "breeds": {
            "Sun Conure": {
                "lifespan": "25-30 years",
                "size": "12 inches",
                "common_issues": ["Fatty liver disease", "Vitamin A deficiency", "Feather destructive behavior"],
                "diet": "Pellets (70%), Fruits/Veggies (20%), Nuts/Seeds (10%)",
                "temperature": "75-85°F (24-29°C)",
                "behavior": "Social, noisy, need interaction",
                "emergency_signs": ["Sudden lethargy", "Difficulty breathing", "Loss of balance"]
            },
            "African Grey": {
                "lifespan": "40-60 years",
                "size": "13 inches",
                "common_issues": ["Calcium deficiency", "Feather picking", "Aspergillosis"],
                "diet": "Calcium-rich pellets, vegetables, limited seeds",
                "temperature": "75-85°F",
                "behavior": "Intelligent, sensitive, prone to stress",
                "emergency_signs": ["Seizures", "Bleeding", "Inability to perch"]
            },
            # Add more breeds as needed
        },
        
        "symptoms": {
            "feather_plucking": {
                "common_causes": ["Stress/anxiety", "Boredom", "Nutritional deficiencies", "Parasites"],
                "diagnostics": ["Skin scrape", "Blood work", "Behavioral assessment"],
                "treatments": ["Environmental enrichment", "Diet improvement", "Medical treatment"]
            },
            "respiratory_distress": {
                "common_causes": ["Bacterial infection", "Fungal infection", "Allergies", "Foreign body"],
                "diagnostics": ["X-ray", "Culture", "Blood gases"],
                "treatments": ["Antibiotics", "Antifungals", "Supportive care"]
            },
            "lethargy": {
                "common_causes": ["Infection", "Metabolic disorder", "Organ failure", "Pain"],
                "diagnostics": ["Blood work", "X-ray", "Physical exam"],
                "treatments": ["Address underlying cause", "Supportive care", "Pain management"]
            }
        }
    }
    
    @classmethod
    def get_breed_info(cls, breed: str) -> Dict:
        """Get information about a specific breed"""
        return cls.CORE_KNOWLEDGE["breeds"].get(breed, {
            "general_info": "General parrot care applies",
            "common_issues": ["Various health concerns"],
            "diet": "Balanced pellets with fresh produce",
            "temperature": "75-85°F"
        })
    
    @classmethod
    def get_symptom_info(cls, symptom_keyword: str) -> Dict:
        """Get information about a symptom"""
        for symptom, info in cls.CORE_KNOWLEDGE["symptoms"].items():
            if symptom_keyword in symptom or symptom in symptom_keyword:
                return info
        return {"common_causes": ["Multiple possible causes"], "treatments": ["General supportive care"]}

# --- RAG SYSTEM ---
class AvianRAGSystem:
    """Robust RAG system for avian medical knowledge"""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.initialize_embeddings()
        self.load_or_create_vector_store()
    
    def initialize_embeddings(self):
        """Initialize embeddings model with fallback"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            print(f"Warning: Could not load HuggingFace embeddings: {e}")
            # Fallback to simpler model
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except:
                self.embeddings = None
    
    def load_or_create_vector_store(self):
        """Load existing vector store or create default one"""
        try:
            if os.path.exists(Config.VECTOR_DB_PATH) and self.embeddings:
                self.vector_store = FAISS.load_local(
                    Config.VECTOR_DB_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                self.create_default_knowledge_base()
        except Exception as e:
            print(f"Warning: Could not load vector store: {e}")
            self.create_default_knowledge_base()
    
    def create_default_knowledge_base(self):
        """Create a default knowledge base with avian medical information"""
        default_documents = [
            "Avian Emergency Signs: Difficulty breathing, bleeding, seizures, inability to perch, no droppings for 12+ hours.",
            "Common Parrot Diseases: Psittacosis, Aspergillosis, PBFD, Polyomavirus, Giardia.",
            "Nutrition Basics: Fresh water daily, balanced pellets, fresh vegetables, limited fruits, avoid avocado/chocolate/caffeine.",
            "Environmental Needs: Temperature 75-85°F, humidity 40-60%, no Teflon/non-stick cookware, no smoking around birds.",
            "Behavioral Health: Social interaction needed, environmental enrichment, regular routine, minimize stress.",
            "First Aid: Stop bleeding with cornstarch, keep warm, offer electrolyte solution, contact avian vet immediately.",
            "Feather Issues: Can indicate stress, nutritional deficiency, parasites, or medical conditions.",
            "Respiratory Issues: Sneezing, nasal discharge, tail bobbing, open-mouth breathing require veterinary attention.",
            "Digestive Problems: Monitor droppings color and consistency. Watery droppings indicate issues.",
            "Avian Veterinary Care: Annual checkups recommended. Blood work, fecal exams important for health monitoring."
        ]
        
        if self.embeddings:
            from langchain.schema import Document
            documents = [Document(page_content=doc) for doc in default_documents]
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.vector_store.save_local(Config.VECTOR_DB_PATH)
    
    def query(self, query: str, k: int = None) -> str:
        """Query the RAG system"""
        if not self.vector_store or not self.embeddings:
            return self.get_fallback_knowledge(query)
        
        try:
            k = k or Config.SIMILARITY_SEARCH_K
            docs = self.vector_store.similarity_search(query, k=k)
            context = "\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            print(f"RAG query error: {e}")
            return self.get_fallback_knowledge(query)
    
    def get_fallback_knowledge(self, query: str) -> str:
        """Get fallback knowledge when RAG fails"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['emergency', 'critical', 'urgent']):
            return "EMERGENCY: Contact avian veterinarian immediately for: difficulty breathing, bleeding, seizures, inability to perch."
        
        elif any(word in query_lower for word in ['feather', 'pluck']):
            return "Feather issues often relate to: stress, nutritional deficiencies, parasites, or medical conditions. Increase enrichment, review diet."
        
        elif any(word in query_lower for word in ['breath', 'sneeze', 'cough']):
            return "Respiratory issues require prompt attention. Keep bird warm, ensure clean air, contact vet if symptoms persist."
        
        elif any(word in query_lower for word in ['eat', 'appetite', 'food']):
            return "Appetite changes: Monitor weight, offer favorite foods, ensure fresh water. If not eating for 24h, seek vet care."
        
        else:
            return "General avian care: Maintain proper temperature (75-85°F), fresh water daily, balanced diet, regular vet checkups."

# --- CREWAI AGENTS ---
class AvianDiagnosticCrew:
    """CrewAI system for avian diagnostics"""
    
    def __init__(self):
        Config.validate()
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=0.7,
            max_tokens=2000
        )
        self.rag_system = AvianRAGSystem()
        self.knowledge_base = AvianKnowledgeBase()
    
    def create_agents(self, breed: str, query: str):
        """Create specialized agents for the diagnostic crew"""
        
        # Get breed-specific context
        breed_info = self.knowledge_base.get_breed_info(breed)
        breed_context = f"""
        Breed: {breed}
        Characteristics: {json.dumps(breed_info, indent=2)}
        """
        
        # 1. Chief Avian Veterinarian
        chief_veterinarian = Agent(
            role="Chief Avian Veterinarian",
            goal=f"Provide comprehensive medical assessment for this {breed}",
            backstory="""You are Dr. Aviana Feathers, board-certified avian veterinarian with 30 years experience.
            You've treated thousands of parrots and published research on psittacine medicine.
            You're known for your thorough differential diagnoses and compassionate care.""",
            llm=self.llm,
            verbose=Config.VERBOSE,
            allow_delegation=True,
            max_iter=Config.MAX_ITERATIONS,
            memory=True
        )
        
        # 2. Avian Behavior Specialist
        behavior_specialist = Agent(
            role="Avian Behaviorist and Ethologist",
            goal="Analyze behavioral and psychological aspects",
            backstory="""You are Dr. Skye Talon, PhD in avian ethology with 25 years experience.
            You specialize in parrot behavior, stress indicators, and environmental enrichment.
            You've worked with zoos, sanctuaries, and private owners worldwide.""",
            llm=self.llm,
            verbose=Config.VERBOSE,
            allow_delegation=True,
            max_iter=Config.MAX_ITERATIONS,
            memory=True
        )
        
        # 3. Avian Nutrition Specialist
        nutrition_specialist = Agent(
            role="Avian Nutritionist",
            goal="Provide species-specific dietary recommendations",
            backstory="""You are Dr. Peregrine Seed, avian nutritionist with 20 years experience.
            You've formulated diets for major aviaries and published on parrot nutritional requirements.
            You specialize in diet-related health issues and supplementation.""",
            llm=self.llm,
            verbose=Config.VERBOSE,
            allow_delegation=True,
            max_iter=Config.MAX_ITERATIONS,
            memory=True
        )
        
        # 4. Emergency Care Coordinator
        emergency_coordinator = Agent(
            role="Emergency Avian Care Coordinator",
            goal="Assess urgency and provide emergency protocols",
            backstory="""You are Dr. Falcon Emergency, emergency avian specialist.
            You've managed critical care units and trained veterinary emergency teams.
            You're expert in triage, stabilization, and emergency procedures for birds.""",
            llm=self.llm,
            verbose=Config.VERBOSE,
            allow_delegation=True,
            max_iter=Config.MAX_ITERATIONS,
            memory=True
        )
        
        return {
            "chief_vet": chief_veterinarian,
            "behaviorist": behavior_specialist,
            "nutritionist": nutrition_specialist,
            "emergency": emergency_coordinator
        }
    
    def create_tasks(self, agents: Dict, breed: str, query: str, context: str):
        """Create specialized tasks for each agent"""
        
        # Task 1: Medical Assessment
        medical_task = Task(
            description=f"""
            Conduct thorough medical assessment for this case:
            
            PATIENT: {breed}
            PRESENTING CONCERN: {query}
            MEDICAL CONTEXT: {context}
            
            Provide:
            1. Differential Diagnosis (ranked by likelihood)
            2. Clinical Signs Analysis
            3. Diagnostic Recommendations
            4. Treatment Options Outline
            
            Be specific and reference current avian medical knowledge.
            """,
            expected_output="Comprehensive medical assessment with specific recommendations",
            agent=agents["chief_vet"],
            output_file="medical_assessment.md"
        )
        
        # Task 2: Behavioral Analysis
        behavioral_task = Task(
            description=f"""
            Analyze behavioral aspects for this case:
            
            BREED: {breed}
            CONCERN: {query}
            
            Assess:
            1. Behavioral red flags
            2. Environmental stressors
            3. Enrichment recommendations
            4. Behavioral modification strategies
            5. Owner education points
            
            Consider breed-specific behavioral tendencies.
            """,
            expected_output="Detailed behavioral analysis and intervention plan",
            agent=agents["behaviorist"],
            output_file="behavioral_analysis.md"
        )
        
        # Task 3: Nutritional Assessment
        nutritional_task = Task(
            description=f"""
            Provide nutritional assessment for this case:
            
            BREED: {breed}
            SYMPTOMS: {query}
            
            Evaluate:
            1. Current diet adequacy
            2. Potential nutritional deficiencies
            3. Species-specific dietary needs
            4. Supplement recommendations
            5. Feeding schedule adjustments
            
            Reference {breed} specific nutritional requirements.
            """,
            expected_output="Species-specific nutritional plan and recommendations",
            agent=agents["nutritionist"],
            output_file="nutritional_assessment.md"
        )
        
        # Task 4: Urgency Assessment
        emergency_task = Task(
            description=f"""
            Assess urgency and provide emergency guidance:
            
            CASE: {breed} with {query}
            
            Determine:
            1. Triage level (Critical/Urgent/Non-urgent)
            2. Immediate stabilization steps
            3. Emergency red flags
            4. When to seek veterinary care
            5. First aid instructions
            
            Be clear about emergency vs non-emergency situations.
            """,
            expected_output="Urgency assessment and emergency action plan",
            agent=agents["emergency"],
            output_file="emergency_assessment.md"
        )
        
        return [medical_task, behavioral_task, nutritional_task, emergency_task]
    
    def execute_diagnostic(self, breed: str, query: str) -> str:
        """Execute the full diagnostic workflow"""
        
        try:
            # Step 1: Get RAG context
            print(f"🔍 Retrieving medical context for: {query[:50]}...")
            rag_context = self.rag_system.query(query)
            
            # Step 2: Create agents
            print("🤖 Initializing specialist agents...")
            agents = self.create_agents(breed, query)
            
            # Step 3: Create tasks
            tasks = self.create_tasks(agents, breed, query, rag_context)
            
            # Step 4: Create and run crew
            print("🚀 Starting diagnostic analysis...")
            diagnostic_crew = Crew(
                agents=list(agents.values()),
                tasks=tasks,
                process=Process.sequential,
                verbose=Config.VERBOSE,
                memory=True,
                max_iter=Config.MAX_ITERATIONS
            )
            
            result = diagnostic_crew.kickoff()
            
            # Step 5: Compile final report
            final_report = self.compile_final_report(str(result), breed, query)
            
            return final_report
            
        except Exception as e:
            print(f"Diagnostic error: {e}")
            return self.generate_fallback_report(breed, query, str(e))
    
    def compile_final_report(self, crew_output: str, breed: str, query: str) -> str:
        """Compile a professional final report from crew output"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# 🦜 SQUAWK-A-THON AVIAN DIAGNOSTIC REPORT
**Generated:** {timestamp}
**Case ID:** {hash(f"{breed}{query}{timestamp}") % 1000000}

## 📋 CASE SUMMARY
**Breed:** {breed}
**Presenting Concern:** {query}
**Assessment Team:** Chief Avian Veterinarian, Behavior Specialist, Nutritionist, Emergency Coordinator

---

## 🩺 COMPREHENSIVE DIAGNOSTIC ANALYSIS

{crew_output}

---

## 🎯 INTEGRATED RECOMMENDATIONS

### 1. Medical Management
- Follow differential diagnosis priorities
- Implement recommended diagnostics
- Monitor response to interventions

### 2. Behavioral Support
- Address environmental factors
- Implement enrichment strategies
- Monitor behavioral changes

### 3. Nutritional Optimization
- Adjust diet as recommended
- Monitor weight and condition
- Consider suggested supplements

### 4. Follow-up Protocol
- Schedule veterinary recheck
- Document progress daily
- Report any changes immediately

---

## ⚠️ IMPORTANT DISCLAIMER
This AI-generated report is for informational purposes only. 
It is NOT a substitute for professional veterinary care.
Always consult with a certified avian veterinarian for diagnosis and treatment.

**Emergency Contact:** Association of Avian Veterinarians - 1-800-AVIAN-VET
**Report Generated by:** Squawk-a-Thon AI Diagnostic System
"""
        
        return report
    
    def generate_fallback_report(self, breed: str, query: str, error: str = "") -> str:
        """Generate fallback report if CrewAI fails"""
        
        # Get basic knowledge
        breed_info = self.knowledge_base.get_breed_info(breed)
        rag_context = self.rag_system.query(query)
        
        return f"""
# 🩺 AVIAN HEALTH ADVISORY - BASIC ASSESSMENT

**Breed:** {breed}
**Concern:** {query}

## 📋 IMMEDIATE GUIDANCE

### For {breed}:
- **Temperature:** Maintain {breed_info.get('temperature', '75-85°F')}
- **Diet:** {breed_info.get('diet', 'Balanced pellets with fresh produce')}
- **Common Issues:** {', '.join(breed_info.get('common_issues', ['Monitor closely']))}

### Based on your concern:
{rag_context}

### 🚨 EMERGENCY PROTOCOL:
1. **Warmth** - Provide 85°F warm area
2. **Hydration** - Offer electrolyte solution
3. **Quiet** - Stress-free environment
4. **Monitoring** - Check every 30 minutes
5. **Veterinary Contact** - Call avian vet immediately if worsening

*Note: Advanced diagnostic system temporarily unavailable. {error if error else 'Please consult avian veterinarian for proper assessment.'}*

**Emergency:** 1-800-AVIAN-VET | **Online Resources:** aav.org
"""

# --- MAIN FUNCTIONS ---
def load_rag_chain(query: str) -> str:
    """
    Load RAG context for a query
    Args:
        query: User's query about avian health
    Returns:
        Context string from knowledge base
    """
    try:
        rag_system = AvianRAGSystem()
        return rag_system.query(query)
    except Exception as e:
        print(f"RAG error: {e}")
        return AvianKnowledgeBase().get_fallback_context(query)

def crew_ai_response(query: str, breed: str) -> str:
    """
    Generate AI response using CrewAI diagnostic system
    Args:
        query: User's health concern
        breed: Bird breed
    Returns:
        Comprehensive diagnostic report
    """
    try:
        # Initialize diagnostic crew
        diagnostic_crew = AvianDiagnosticCrew()
        
        # Execute diagnostic workflow
        print(f"Starting diagnostic for {breed}: {query[:50]}...")
        report = diagnostic_crew.execute_diagnostic(breed, query)
        
        return report
        
    except Exception as e:
        print(f"CrewAI error: {e}")
        
        # Create fallback response
        fallback_response = f"""
        # 🦜 Squawk-a-Thon Diagnostic System
        
        **Breed:** {breed}
        **Concern:** {query}
        
        ## ⚠️ System Status
        The advanced diagnostic system is experiencing technical difficulties.
        
        ## 📋 Basic Guidance for {breed}:
        1. **Monitor closely** for changes in behavior
        2. **Ensure basic needs:** Fresh water, proper temperature, balanced diet
        3. **Reduce stress:** Quiet environment, familiar routine
        4. **Contact avian veterinarian** if symptoms persist or worsen
        
        ## 🚨 Emergency Signs (Seek Immediate Vet Care):
        - Difficulty breathing
        - Bleeding that doesn't stop
        - Inability to perch or stand
        - Seizures or loss of consciousness
        
        *Error details: {str(e)[:100]}*
        """
        
        return fallback_response

# --- TESTING FUNCTION ---
def test_system():
    """Test the diagnostic system with sample queries"""
    test_cases = [
        ("Sun Conure", "Feather plucking and lethargy for 3 days"),
        ("African Grey", "Sneezing with nasal discharge"),
        ("Cockatiel", "Not eating, sitting at bottom of cage"),
        ("Macaw", "Aggressive behavior and feather fluffing")
    ]
    
    for breed, query in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {breed} - {query}")
        print(f"{'='*60}")
        
        # Test RAG
        rag_result = load_rag_chain(query)
        print(f"RAG Result (first 200 chars): {rag_result[:200]}...")
        
        # Test CrewAI (optional - can be slow)
        # crew_result = crew_ai_response(query, breed)
        # print(f"CrewAI Response (first 300 chars): {crew_result[:300]}...")
        
        print(f"{'='*60}")

if __name__ == "__main__":
    # Validate configuration
    try:
        Config.validate()
        print("✅ Configuration validated")
        
        # Test the system
        test_system()
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("Please ensure OPENAI_API_KEY is set in .env file")
