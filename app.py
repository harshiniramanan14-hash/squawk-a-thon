import streamlit as st
import os
import google.generativeai as genai
from brain import load_rag_chain, crew_ai_response
from PIL import Image
import tempfile

# --- 1. KEY VALIDATION (Fixes TypeError) ---
from dotenv import load_dotenv
load_dotenv()

google_key = os.getenv("GOOGLE_API_KEY")
if not google_key:
    st.error("🚨 GOOGLE_API_KEY not found in .env!")
    st.stop()
genai.configure(api_key=google_key)

# --- 2. RAINFOREST UI ---
st.set_page_config(page_title="Squawk-A-Thon 🦜", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                url('https://images.unsplash.com/photo-1552084117-56a987666449?q=80&w=2000');
    background-size: cover;
    color: #e8f5e9;
}
.result-box { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); padding: 20px; border-radius: 15px; border: 1px solid #4ade80; }
</style>
""", unsafe_allow_html=True)

# Logo Loading - Fixed Filename
logo_path = "assets/ChatGPT Image Jan 23, 2026, 07_37_38 PM.jpg" # Note the .jpg vs .png in your screenshot
try:
    st.image(logo_path, width=150)
except:
    st.warning("Logo not found in assets/ folder.")

st.title("🌿 Squawk-A-Thon")
st.write("### The Rainforest Multimodal Avian Assistant")

# --- 3. INPUTS ---
col1, col2 = st.columns(2)
with col1:
    breed = st.selectbox("Bird Breed", ["Sun Conure", "Jenday Conure", "African Grey", "Cockatiel", "Budgerigar", "Macaw"])
    query = st.text_area("What is the concern?", height=150)
with col2:
    st.write("📷 **Upload Evidence**")
    media = st.file_uploader("Upload Video or Audio", type=["mp4", "mp3", "wav", "mov"])

# --- 4. EXECUTION ---
if st.button("RUN MULTIMODAL DIAGNOSTIC 🌲"):
    with st.spinner("The Rainforest Spirits are analyzing..."):
        # Text/RAG Analysis
        rag_chain = load_rag_chain()
        rag_context = rag_chain.invoke(query)['result']
        crew_out = crew_ai_response(f"{query} \nContext: {rag_context}", breed)

        # Multimodal Analysis (Gemini)
        media_out = ""
        if media:
            model = genai.GenerativeModel('gemini-1.5-flash')
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(media.name)[1]) as tmp:
                tmp.write(media.read())
                gen_file = genai.upload_file(tmp.name)
                media_out = model.generate_content([gen_file, f"Analyze this media for a {breed} showing {query}."]).text

        # Results Display
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader("📋 Final Avian Specialist Report")
        st.write(crew_out)
        if media_out:
            st.markdown("---")
            st.write("**🎬 Media-Based Observation:**")
            st.write(media_out)
        st.markdown('</div>', unsafe_allow_html=True)
