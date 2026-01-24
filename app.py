import streamlit as st
import os
import google.generativeai as genai
from brain import load_rag_chain, crew_ai_response
from PIL import Image
import tempfile
from dotenv import load_dotenv

load_dotenv()

# --- 1. SAFE KEY LOADING (Fixes TypeError & 401) ---
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not GOOGLE_KEY or not OPENAI_KEY:
    st.error("🚨 Missing API Keys in .env! Ensure 'GOOGLE_API_KEY' and 'OPENAI_API_KEY' are set.")
    st.stop()

genai.configure(api_key=GOOGLE_KEY)

# --- 2. RAINFOREST UI ---
st.set_page_config(page_title="Squawk-A-Thon 🦜", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                url('https://images.unsplash.com/photo-1590273466070-40c466b4432d?q=80&w=2000');
    background-size: cover; color: #f0fdf4;
}
.main-title { color: #4ade80; text-align: center; font-size: 3.5rem; text-shadow: 2px 2px #064e3b; margin-top: -60px; }
.result-card { background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(15px); padding: 25px; border-radius: 20px; border: 1px solid #4ade80; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGO LOADING (Fixes 'Logo not found') ---
col1, col2, col3 = st.columns([1,2,1])
with col2:
    # Exact filename provided in your prompt
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    st.markdown('<h1 class="main-title">🌿 Squawk-A-Thon</h1>', unsafe_allow_html=True)

# --- 4. MULTIMODAL DIAGNOSTIC ---
st.write("### 🏥 Specialist Avian Consultation")
breed = st.selectbox("Select Breed:", ["Sun Conure", "Jenday Conure", "Macaw", "African Grey", "Cockatiel", "Budgie","cockatoo","other"])
query = st.text_area("Describe the concern (e.g., lethargy, feather plucking, respiratory sound):")

# Feature: Picture, Audio, and Video Support
media_file = st.file_uploader("📷 Upload Evidence (Photo/Video/Audio)", type=["jpg", "jpeg", "png", "mp4", "mp3", "mov"])

if media_file:
    ext = os.path.splitext(media_file.name)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]: st.image(media_file, width=400)
    elif ext in [".mp4", ".mov"]: st.video(media_file)
    elif ext in [".mp3", ".wav"]: st.audio(media_file)

# --- 5. EXECUTION ---
if st.button("RUN MULTIMODAL DIAGNOSTIC 🌲"):
    with st.spinner("Analyzing the rainforest echoes..."):
        try:
            # RAG + OpenAI CrewAI Reasoning
            context = load_rag_chain(query)
            report = crew_ai_response(f"{query}\n\nContext: {context}", breed)

            # Vision/Audio Specialist Findings (Gemini 1.5 Flash)
            vision_out = ""
            if media_file:
                model = genai.GenerativeModel('gemini-1.5-flash')
                if ext in [".jpg", ".jpeg", ".png"]:
                    img = Image.open(media_file)
                    vision_out = model.generate_content([f"Analyze this {breed} health image: {query}", img]).text
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(media_file.read())
                        gen_f = genai.upload_file(tmp.name)
                        vision_out = model.generate_content([gen_f, f"Analyze this media for {breed} health: {query}"]).text

            # --- DISPLAY RESULTS ---
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("📋 Specialist Diagnostic Report")
            st.write(report)
            if vision_out:
                st.markdown("---")
                st.write("**🎬 Vision/Audio Observations:**")
                st.write(vision_out)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Analysis Failed: {e}")

