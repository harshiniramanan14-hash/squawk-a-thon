import streamlit as st
import os
import google.generativeai as genai
from brain import AvianSpecialistCrew, update_db
from PIL import Image
import tempfile

# --- 1. RAINFOREST STYLING ---
st.set_page_config(page_title="Squawk-A-Thon AI", page_icon="🦜", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                    url('https://images.unsplash.com/photo-1590273466070-40c466b4432d?q=80&w=2000');
        background-size: cover;
        color: #f0fdf4;
    }
    .main-title { color: #4ade80; text-align: center; font-size: 3.5rem; text-shadow: 2px 2px #064e3b; }
    .stSelectbox div, .stTextArea textarea { background-color: rgba(6, 78, 59, 0.8) !important; color: white !important; border: 1px solid #4ade80 !important; }
    .result-card { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(12px); padding: 25px; border-radius: 20px; border: 1px solid #fbbf24; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGO & HEADER ---
col1, col2, col3 = st.columns([1,2,1])
with col2:
    logo_file = "ChatGPT Image Jan 23, 2026, 07_37_38 PM.jpg" # Fixed extension
    if os.path.exists(logo_file):
        st.image(logo_file, use_container_width=True)
    st.markdown('<h1 class="main-title">🌿 Squawk-A-Thon</h1>', unsafe_allow_html=True)

# --- 3. MULTIMODAL & RAG LOGIC ---
st.sidebar.header("📁 Knowledge Hub")
clinical_docs = st.sidebar.file_uploader("Upload Clinical/Behavioral Docs", accept_multiple_files=True)
if st.sidebar.button("Update Knowledge Base"):
    texts = [doc.read().decode("utf-8") for doc in clinical_docs if doc.name.endswith('.txt')]
    if texts:
        update_db(texts)
        st.sidebar.success("RAG Synced!")

# --- 4. DIAGNOSTIC CENTER ---
st.write("### 🏥 Specialist Consultation")
breed = st.selectbox("Select Breed:", ["Sun Conure", "Jenday Conure", "Macaw", "African Grey", "Cockatiel", "Budgie","cockatoo","other"])
symptoms = st.text_area("Describe Symptoms (e.g., Lethargy, tail bobbing)")
user_media = st.file_uploader("📷 Upload Audio/Video Evidence", type=["mp4", "mp3", "wav"])

if st.button("RUN DEEP DIAGNOSTIC 🌲"):
    crew = AvianSpecialistCrew()
    with st.spinner("Analyzing rainforest echoes..."):
        # Text-based RAG Analysis
        report = crew.run_diagnostic(breed, symptoms)
        
        # Audio/Video Analysis with Gemini
        media_analysis = ""
        if user_media:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel('gemini-1.5-flash')
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(user_media.name)[1]) as tmp:
                tmp.write(user_media.read())
                gen_file = genai.upload_file(tmp.name)
                # Wait for file processing
                media_analysis = model.generate_content([gen_file, f"Analyze this {user_media.name} for health issues in a {breed}."]).text
        
        # DISPLAY RESULTS
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("📋 Final Avian Report")
        st.markdown(f"**Agent Consultation:**\n{report}")
        if media_analysis:
            st.markdown(f"--- \n**Media Analysis:**\n{media_analysis}")
        st.markdown('</div>', unsafe_allow_html=True)

st.caption("Educational tool only. Consult a veterinarian for medical emergencies.")
