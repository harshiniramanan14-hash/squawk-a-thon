import os
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Securely load keys to prevent TypeError
groq_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=groq_key)
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

class AvianSpecialistCrew:
    def __init__(self, db_path="avian_knowledge_db"):
        self.db_path = db_path
        
    def get_context(self, query):
        if os.path.exists(self.db_path):
            db = FAISS.load_local(self.db_path, hf_embeddings, allow_dangerous_deserialization=True)
            docs = db.similarity_search(query, k=3)
            return "\n".join([d.page_content for d in docs])
        return "No clinical documents found. Using general knowledge."

    def run_diagnostic(self, breed, symptoms):
        context = self.get_context(f"{breed} {symptoms}")
        
        # Agent 1: Clinical Pathologist
        pathologist = Agent(
            role='Avian Pathologist',
            goal=f'Analyze symptoms for a {breed} based on context: {context}',
            backstory='Expert in avian diseases and clinical markers.',
            llm=llm,
            verbose=True
        )

        # Agent 2: Behaviorist 
        behaviorist = Agent(
            role='Avian Ethologist',
            goal='Interpret behavioral anomalies and stress signals.',
            backstory='Specialist in parrot psychology and environmental stress.',
            llm=llm,
            verbose=True
        )

        task = Task(
            description=f"Analyze the {breed} showing {symptoms}. Cross-reference with clinical data.",
            expected_output="A structured report: Potential Issues, Behavioral Root, and 3 Recovery Steps.",
            agent=pathologist
        )

        crew = Crew(agents=[pathologist, behaviorist], tasks=[task], process=Process.sequential)
        return crew.kickoff()

def update_db(texts, db_path="avian_knowledge_db"):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents(texts)
    if os.path.exists(db_path):
        db = FAISS.load_local(db_path, hf_embeddings, allow_dangerous_deserialization=True)
        db.add_documents(docs)
    else:
        db = FAISS.from_documents(docs, hf_embeddings)
    db.save_local(db_path)
