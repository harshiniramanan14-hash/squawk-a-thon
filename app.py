import streamlit as st
import os
import google.generativeai as genai
from brain import load_rag_chain, crew_ai_response
from PIL import Image
import tempfile
from dotenv import load_dotenv

# --- 1. CONFIG & SECURITY ---
load_dotenv()
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")

# Check keys immediately to prevent TypeErrors
if not GOOGLE_KEY or not GROQ_KEY:
    st.error("🚨 API Keys missing! Ensure GOOGLE_API_KEY and GROQ_API_KEY are in your .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_KEY)

st.set_page_config(page_title="Squawk-A-Thon 🦜", layout="wide")

# --- 2. RAINFOREST CSS ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                url('https://images.unsplash.com/photo-1590273466070-40c466b4432d?q=80&w=2000');
    background-size: cover;
    color: #f0fdf4;
}
.main-title { color: #4ade80; text-align: center; font-size: 3.5rem; text-shadow: 2px 2px #064e3b; margin-top: -60px; }
.result-card { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(12px); padding: 25px; border-radius: 20px; border: 1px solid #4ade80; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGO & HEADER ---
col1, col2, col3 = st.columns([1,2,1])
with col2:
    # Updated to the correct .jpg extension seen in your file list
    logo_path = "logo.jpg"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.info("💡 Place 'ChatGPT Image Jan 23, 2026, 07_37_38 PM.jpg' in the main folder.")
    st.markdown('<h1 class="main-title">🌿 Squawk-A-Thon</h1>', unsafe_allow_html=True)

# --- 4. MULTIMODAL INPUTS ---
st.write("### 🏥 Specialist Diagnostic Center")
breed = st.selectbox("Select Breed:", ["Sun Conure", "Jenday Conure", "Macaw", "African Grey", "Cockatiel", "Budgie"])
query = st.text_area("Describe the concern (e.g., lethargy, skin rash):")

# Updated Feature: Upload Pictures, Audio, or Video
st.write("### 📷 Evidence Upload (Image/Audio/Video)")
media_file = st.file_uploader("Upload media for AI Vision analysis", type=["jpg", "png", "jpeg", "mp4", "mp3", "mov"])

# Preview media if uploaded
if media_file:
    ext = os.path.splitext(media_file.name)[1].lower()
    if ext in [".jpg", ".png", ".jpeg"]:
        st.image(media_file, caption="Uploaded Image Evidence", width=400)
    elif ext in [".mp4", ".mov"]:
        st.video(media_file)
    elif ext in [".mp3", ".wav"]:
        st.audio(media_file)

# --- 5. EXECUTION ---
if st.button("RUN MULTIMODAL DIAGNOSTIC 🌲"):
    if not query:
        st.warning("Please describe the symptoms first!")
    else:
        with st.spinner("Analyzing the rainforest echoes..."):
            try:
                # 1. RAG + CrewAI Analysis
                context = load_rag_chain(query)
                report = crew_ai_response(f"{query}\n\nContext: {context}", breed)

                # 2. Multimodal Vision Analysis (Gemini)
                vision_analysis = ""
                if media_file:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    # Handle Image vs Video/Audio differently for Gemini
                    if ext in [".jpg", ".png", ".jpeg"]:
                        img = Image.open(media_file)
                        response = model.generate_content([f"Analyze this avian health image for a {breed}: {query}", img])
                        vision_analysis = response.text
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            tmp.write(media_file.read())
                            gen_file = genai.upload_file(tmp.name)
                            vision_analysis = model.generate_content([gen_file, f"Analyze this media for a {breed}: {query}"]).text

                # --- DISPLAY RESULTS ---
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.subheader("📋 Final Avian Health Report")
                st.markdown(f"**Specialist Board Consultation:**\n\n{report}")
                if vision_analysis:
                    st.markdown("---")
                    st.markdown(f"**AI Vision/Audio Findings:**\n\n{vision_analysis}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error: {e}. Check your API keys and internet connection.")

st.caption("Educational tool only. Consult a veterinarian for medical emergencies.")

