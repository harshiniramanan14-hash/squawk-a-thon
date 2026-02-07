import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import time
from brain import load_rag_chain, crew_ai_response

load_dotenv()

# --- INITIALIZATION ---
google_key = os.getenv("GOOGLE_API_KEY")
if google_key:
    genai.configure(api_key=google_key)

st.set_page_config(page_title="Squawk-a-Thon CrewAI 🦜", layout="wide")

# --- CUSTOM CSS (Your Design) ---
st.markdown("""
<style>
.stApp { background: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)), url('https://images.unsplash.com/photo-1552084117-56a987666449?q=80&w=2000'); background-size: cover; color: #f0fdf4; }
.main-title { color: #4ade80; text-align: center; font-size: 3.5rem; text-shadow: 2px 2px #064e3b; margin-top: -40px; }
.result-card { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); padding: 25px; border-radius: 20px; border: 1px solid #4ade80; margin: 20px 0; color: white; }
.agent-card { background: rgba(30, 60, 90, 0.4); padding: 15px; border-radius: 10px; border-left: 4px solid #4ade80; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<h1 class="main-title">🦜 Squawk-a-Thon CrewAI</h1>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>AI-Powered Avian Diagnostic Terminal</h3>", unsafe_allow_html=True)

# --- MULTIMODAL FUNCTION (From Friend's Code) ---
def analyze_multimodal(uploaded_file, user_query, bird_species):
    # Use standard models to avoid 404s
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    file_ext = uploaded_file.name.split('.')[-1].lower()
    temp_path = f"temp_file.{file_ext}"
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        if file_ext in ['jpg', 'jpeg', 'png', 'webp']:
            media_input = Image.open(temp_path)
        else:
            # For Video/Audio, we MUST wait for the 'ACTIVE' state
            st.info("📤 Uploading to Google AI...")
            genai_file = genai.upload_file(path=temp_path)
            
            # CRITICAL: Polling loop to wait for processing
            while genai_file.state.name == "PROCESSING":
                time.sleep(2)
                genai_file = genai.get_file(genai_file.name)
            
            if genai_file.state.name != "ACTIVE":
                return f"❌ Media processing failed: {genai_file.state.name}"
            
            media_input = genai_file

        # Generate content with a specific timeout to prevent the crash in your image
        response = model.generate_content(
            [media_input, f"Diagnose this {bird_species}: {user_query}"],
            request_options={"timeout": 600} # Increased timeout to 10 minutes
        )
        return response.text

    except Exception as e:
        return f"❌ Multimodal Error: {str(e)}"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
# --- MAIN UI ---
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### 📝 Case Input")
    breed = st.selectbox("Breed:", ["Sun Conure", "African Grey", "Cockatiel", "Budgie", "Macaw", "Other"])
    severity = st.select_slider("Severity:", options=["Mild", "Moderate", "Severe", "Critical"])
    query = st.text_area("Symptoms:", height=150, placeholder="Describe behavior, appetite, droppings...")
    media_file = st.file_uploader("📷 Evidence (Image/Video/Audio):", type=['jpg','png','mp4','mp3','wav'])

with col2:
    st.markdown("### 📊 Analysis Engine")
    if st.button("🚀 LAUNCH MULTI-AGENT DIAGNOSIS", type="primary", use_container_width=True):
        if not query or not media_file:
            st.warning("Please provide both symptoms and media evidence.")
        else:
            # 1. RAG Check
            rag_context = load_rag_chain(query)
            
            # 2. Parallel Processing
            with st.spinner("🤖 CrewAI Agents & Gemini Vision analyzing..."):
                # CrewAI Response
                diagnosis = crew_ai_response(query, breed)
                # Gemini Vision Response
                media_analysis = analyze_multimodal(media_file, query, breed)
            
            # --- DISPLAY RESULTS ---
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 📋 DIAGNOSTIC REPORT")
            
            st.markdown(f"**Specialist Consensus:**\n{diagnosis}")
            st.markdown("---")
            st.markdown(f"**Visual Observations:**\n{media_analysis}")
            
            st.info(f"💡 **RAG Knowledge Base Note:** {rag_context}")
            st.markdown('</div>', unsafe_allow_html=True)

# --- SIDEBAR SYSTEM STATUS ---
with st.sidebar:
    st.header("🔧 System Status")
    st.success("Groq LLM: Connected")
    st.success("Gemini Vision: Active")
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** This tool is for educational purposes. Consult a vet for emergencies.")

