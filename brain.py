import os
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# 1. LLM & Embeddings Setup
llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

class AvianSpecialistCrew:
    def __init__(self, vector_store_path="avian_knowledge_db"):
        self.vector_store_path = vector_store_path
        
    def get_retriever_context(self, query):
        if os.path.exists(self.vector_store_path):
            db = FAISS.load_local(self.vector_store_path, hf_embeddings, allow_dangerous_deserialization=True)
            docs = db.similarity_search(query, k=3)
            return "\n".join([d.page_content for d in docs])
        return "No specific documents found in the database."

    def run_diagnostic(self, breed, symptoms):
        context = self.get_retriever_context(f"{breed} {symptoms}")
        
        # Agent 1: The Avian Pathologist
        pathologist = Agent(
            role='Senior Avian Pathologist',
            goal=f'Analyze {symptoms} for a {breed} using the provided database context.',
            backstory='Expert in avian clinical signs and behavior. You translate data into health risks.',
            llm=llm
        )

        # Agent 2: The Behaviorist
        behaviorist = Agent(
            role='Avian Ethologist',
            goal='Interpret behavioral changes and suggest environmental adjustments.',
            backstory='Specialist in bird psychology and flock dynamics.',
            llm=llm
        )

        task = Task(
            description=f"Context: {context}\nAnalyze this {breed}: {symptoms}. Provide a structured diagnosis.",
            expected_output="A summary of health risks, behavioral insights, and 3 immediate care steps.",
            agent=pathologist
        )

        crew = Crew(agents=[pathologist, behaviorist], tasks=[task], process=Process.sequential)
        return crew.kickoff()

def update_knowledge_base(text_list, db_path="avian_knowledge_db"):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(text_list)
    if os.path.exists(db_path):
        db = FAISS.load_local(db_path, hf_embeddings, allow_dangerous_deserialization=True)
        db.add_documents(docs)
    else:
        db = FAISS.from_documents(docs, hf_embeddings)
    db.save_local(db_path)
