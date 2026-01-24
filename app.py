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

if not GOOGLE_KEY:
    st.error("🚨 GOOGLE_API_KEY missing from .env!")
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
.stSidebar { background-color: rgba(6, 78, 59, 0.5); }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGO LOADING ---
# If your logo isn't appearing, ensure the filename is EXACT or rename it to logo.png
LOGO_FILENAME = "ChatGPT Image Jan 23, 2026, 07_37_38 PM.png"

col1, col2, col3 = st.columns([1,2,1])
with col2:
    if os.path.exists(LOGO_FILENAME):
        st.image(LOGO_FILENAME, use_container_width=True)
    else:
        st.warning(f"⚠️ Logo not found. Ensure '{LOGO_FILENAME}' is in the main folder.")
    st.markdown('<h1 class="main-title">🌿 Squawk-A-Thon</h1>', unsafe_allow_html=True)

# --- 4. INPUTS ---
st.write("### 🏥 Specialist Diagnostic Center")
breed = st.selectbox("Select Breed:", ["Sun Conure", "Jenday Conure", "Macaw", "African Grey", "Cockatiel", "Budgie"])
query = st.text_area("Describe the concern (e.g., lethargy, tail bobbing):")

# Media Uploads
st.write("### 📷 Evidence Upload")
media_file = st.file_uploader("Upload Video (Behavior) or Audio (Sounds)", type=["mp4", "mp3", "wav", "mov"])

# --- 5. MEDIA PREVIEW (NEW) ---
if media_file:
    file_extension = os.path.splitext(media_file.name)[1].lower()
    if file_extension in [".mp4", ".mov"]:
        st.video(media_file)
    elif file_extension in [".mp3", ".wav"]:
        st.audio(media_file)

# --- 6. EXECUTION ---
if st.button("RUN MULTIMODAL DIAGNOSTIC 🌲"):
    if not query:
        st.warning("Please describe the concern first.")
    else:
        with st.spinner("Analyzing the rainforest echoes..."):
            try:
                # 1. RAG + Crew Analysis
                # Note: Ensure brain.py is in the same folder
                context = load_rag_chain(query)
                report = crew_ai_response(f"{query}\n\nContext: {context}", breed)

                # 2. Multimodal Media Analysis
                media_analysis = ""
                if media_file:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                        tmp.write(media_file.read())
                        gen_file = genai.upload_file(tmp.name)
                        media_analysis = model.generate_content([gen_file, f"Analyze this media for a {breed} showing {query}."]).text

                # --- DISPLAY RESULTS ---
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.subheader("📋 Final Avian Health Report")
                st.markdown(f"**Specialist Board Consultation:**\n\n{report}")
                if media_analysis:
                    st.markdown("---")
                    st.markdown(f"**AI Vision/Audio Findings:**\n\n{media_analysis}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Logic Error: {e}. Check if brain.py is correctly set up.")

st.caption("Educational tool only. Consult a veterinarian for medical emergencies.")
