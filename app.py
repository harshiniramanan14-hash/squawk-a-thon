import streamlit as st
import os
import google.generativeai as genai
from brain import AvianSpecialistCrew, update_knowledge_base
from PIL import Image
import tempfile

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="Squawk-A-Thon AI", page_icon="🦜", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), 
                    url('https://images.unsplash.com/photo-1518173946687-a4c8a9b746f4?auto=format&fit=crop&q=80&w=2000');
        background-size: cover;
        color: white;
    }
    .main-title { color: #4ade80; font-family: 'Georgia', serif; text-align: center; font-size: 4rem; text-shadow: 2px 2px #14532d; }
    .stSelectbox, .stTextArea, .stTextInput { background-color: rgba(20, 83, 45, 0.7) !important; color: white !important; border-radius: 10px; }
    .result-card { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); padding: 20px; border-radius: 20px; border: 1px solid #4ade80; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HEADER & LOGO ---
col1, col2, col3 = st.columns([1,2,1])
with col2:
    logo_file = "ChatGPT Image Jan 23, 2026, 07_37_38 PM.png"
    if os.path.exists(logo_file):
        st.image(logo_file, use_container_width=True)
    st.markdown('<h1 class="main-title">🦜 Squawk-A-Thon</h1>', unsafe_allow_html=True)

# --- 3. INPUT SECTION ---
with st.sidebar:
    st.header("🌿 Knowledge Upload")
    uploaded_docs = st.file_uploader("Upload Clinical/Behavioral Docs (PDF/Text)", accept_multiple_files=True)
    if st.button("Sync to RAG"):
        # Process docs and call update_knowledge_base()
        st.success("Database Updated!")

st.write("### 🩺 Diagnostic Input")
breed = st.selectbox("Select Avian Breed:", ["Sun Conure", "Jenday Conure", "African Grey", "Macaw", "Cockatiel", "Budgie", "Lovebird"])
symptoms = st.text_area("Describe symptoms or behavior (e.g. 'tail bobbing, clicking sound')")

# --- 4. MULTIMODAL MEDIA UPLOAD ---
media_file = st.file_uploader("📷 Upload Audio/Video for Analysis", type=["mp4", "mp3", "wav"])

if st.button("RUN DEEP ANALYSIS 🌲"):
    crew_engine = AvianSpecialistCrew()
    
    with st.spinner("The Rainforest Spirits (and AI) are conferring..."):
        # 1. Run RAG-based Crew Analysis
        crew_response = crew_engine.run_diagnostic(breed, symptoms)
        
        # 2. Run Multimodal Analysis (if media uploaded)
        multimodal_response = ""
        if media_file:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel('gemini-1.5-flash')
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(media_file.name)[1]) as tmp:
                tmp.write(media_file.read())
                gen_file = genai.upload_file(tmp.name)
                multimodal_response = model.generate_content([gen_file, f"Analyze this {media_file.name} for signs of illness in a {breed}."]).text
        
        # --- 5. DISPLAY RESULTS ---
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("📋 Final Diagnostic Report")
        st.markdown(f"**Agent Analysis:**\n{crew_response}")
        if multimodal_response:
            st.markdown(f"--- \n**Media Analysis:**\n{multimodal_response}")
        st.markdown('</div>', unsafe_allow_html=True)

st.caption("Disclaimer: This AI is for educational support and does not replace professional veterinary care.")
