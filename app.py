import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# --- INITIALIZATION ---
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- PAGE CONFIG ---
st.set_page_config(page_title="Squawk-A-Thon AI", page_icon="🦜", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f0fdf4 0%, #fffbeb 100%); }
    .title-text { color: #166534; font-weight: 800; text-align: center; font-size: 3rem; }
    .result-card { background-color: white; padding: 2rem; border-radius: 20px; border-left: 10px solid #fbbf24; box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
    div.stButton > button { background: #166534; color: white; border-radius: 12px; width: 100%; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER & LOGO ---
st.markdown('<h1 class="title-text">🦜 Squawk-A-Thon</h1>', unsafe_allow_html=True)

# Loading your specific logo name
try:
    logo = Image.open("ChatGPT Image Jan 23, 2026, 07_37_38 PM.png")
    st.image(logo, use_container_width=True)
except:
    st.warning("Logo file not found. Ensure the filename matches exactly.")

# --- MULTIMODAL DIAGNOSIS LOGIC ---
def analyze_multimodal(uploaded_file, user_query, file_type):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    if file_type == "video":
        # Save temp file for Gemini to process
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        video_file = genai.upload_file(path="temp_video.mp4")
        response = model.generate_content([video_file, user_query])
    elif file_type == "audio":
        with open("temp_audio.mp3", "wb") as f:
            f.write(uploaded_file.getbuffer())
        audio_file = genai.upload_file(path="temp_audio.mp3")
        response = model.generate_content([audio_file, user_query])
    else:
        response = model.generate_content(user_query)
        
    return response.text

# --- UI LAYOUT ---
st.write("### 🏥 Avian Diagnostic Center")
tab1, tab2 = st.tabs(["Upload Media", "Knowledge Base"])

with tab1:
    media_file = st.file_uploader("Upload Bird Video (.mp4) or Audio (.mp3)", type=["mp4", "mp3", "wav"])
    query = st.text_input("What is your concern?", placeholder="e.g., Is my bird's breathing heavy in this video?")
    
    if st.button("START AI ANALYSIS 🚀"):
        if query:
            with st.spinner("Analyzing media... This takes a moment for video."):
                ftype = "video" if media_file and media_file.name.endswith("mp4") else "audio" if media_file else "text"
                result = analyze_multimodal(media_file, query, ftype)
                
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("### 📋 AI Specialist Findings")
                st.write(result)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Please provide a question!")

with tab2:
    st.write("### 📚 Pre-loaded Knowledge Base")
    st.info("The system is pre-trained on the 'Avian Health Essentials' database including respiratory distress sounds and lethargy visual markers.")
    # Here you can list specific folders or files your team added to the GitHub
    st.markdown("""
    - **Resource 1:** Conure Vocalization Patterns (.mp3)
    - **Resource 2:** Recognizing Egg Binding in Parrots (.mp4)
    - **Resource 3:** General Avian Nutrition Guide (.pdf)
    """)
