import streamlit as st
import os
import google.generativeai as genai
from brain import load_rag_chain, crew_ai_response
from PIL import Image
import tempfile
from dotenv import load_dotenv

load_dotenv()

# --- 1. STARTUP VALIDATION (Fixes TypeError) ---
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not GOOGLE_KEY or not OPENAI_KEY:
    st.error("🚨 Missing API Keys! Ensure GOOGLE_API_KEY and OPENAI_API_KEY are in your .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_KEY)

# --- 2. RAINFOREST UI ---
st.set_page_config(page_title="Squawk-a-Thon 🦜", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                url('https://images.unsplash.com/photo-1552084117-56a987666449?q=80&w=2000');
    background-size: cover; color: #f0fdf4;
}
.main-title { color: #4ade80; text-align: center; font-size: 3.5rem; text-shadow: 2px 2px #064e3b; margin-top: -60px; }
.result-card { background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(15px); padding: 25px; border-radius: 20px; border: 1px solid #4ade80; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGO & HEADER ---
col1, col2, col3 = st.columns([1,2,1])
with col2:
    # Fixed Filename logic from your upload
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    st.markdown('<h1 class="main-title">🌿 Squawk-a-Thon</h1>', unsafe_allow_html=True)

# --- 4. MULTIMODAL INPUTS ---
st.write("### 🏥 Specialist Avian Diagnostic Center")
breed = st.selectbox("Select Breed:", ["Sun Conure", "Jenday Conure", "Macaw", "African Grey", "Cockatiel", "Budgie"])
query = st.text_area("Describe the concern (e.g., feather plucking, lethargy, wing drooping):")

media_file = st.file_uploader("📷 Upload Evidence (Photo/Video/Audio)", type=["jpg", "jpeg", "png", "mp4", "mp3", "mov"])

if media_file:
    ext = os.path.splitext(media_file.name)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]: 
        st.image(media_file, width=400)
    elif ext in [".mp4", ".mov"]: 
        st.video(media_file)
    elif ext in [".mp3", ".wav"]: 
        st.audio(media_file)

# --- 5. EXECUTION ---
if st.button("RUN MULTIMODAL DIAGNOSTIC 🌲"):
    with st.spinner("Consulting the Rainforest Spirits..."):
        try:
            # 1. RAG context
            context = load_rag_chain(query)
            
            # 2. CrewAI Analysis (with fallback)
            try:
                report = crew_ai_response(f"{query}\n\nContext: {context}", breed)
            except Exception as crew_error:
                report = f"""📋 **Avian Health Assessment for {breed}**
                
                **Primary Concern:** {query}
                
                **General Guidance:**
                • **Temperature:** Maintain 75-85°F (24-29°C)
                • **Hydration:** Ensure fresh water is always available
                • **Nutrition:** Species-appropriate pellets, fresh veggies
                • **Observation:** Monitor droppings, appetite, activity level
                
                **When to Seek Immediate Vet Care:**
                ‣ Bleeding that doesn't stop in 5 minutes
                ‣ Difficulty breathing
                ‣ Inability to perch or stand
                ‣ No droppings for 12+ hours
                
                *Note: AI system experiencing technical issues. Please consult an avian veterinarian.*"""
            
            # 3. Vision/Audio Analysis
            vision_out = ""
            if media_file:
                try:
                    # Try the newest model first
                    model = genai.GenerativeModel('gemini-1.5-pro')
                except:
                    # Fallback to more available models
                    model = genai.GenerativeModel('gemini-pro')
                
                try:
                    if ext in [".jpg", ".jpeg", ".png"]:
                        img = Image.open(media_file)
                        vision_out = model.generate_content([
                            f"Analyze this {breed} parrot's health condition. Look for: feather condition, posture, eyes, beak. Concern: {query}",
                            img
                        ]).text
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            tmp.write(media_file.read())
                            gen_f = genai.upload_file(tmp.name)
                            vision_out = model.generate_content([
                                f"Analyze this media for {breed} parrot health indicators. Concern: {query}",
                                gen_f
                            ]).text
                except Exception as vision_error:
                    vision_out = f"Media analysis unavailable: {str(vision_error)[:100]}"
            
            # --- DISPLAY RESULTS ---
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("📋 Specialist Diagnostic Report")
            st.write(report)
            
            if vision_out:
                st.markdown("---")
                st.subheader("🎬 Vision/Audio Specialist Findings")
                st.write(vision_out)
                
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error("⚠️ Analysis System Busy")
            st.info(f"""
            **Immediate Steps for {breed}:**
            1. **Isolate** if showing signs of illness
            2. **Warmth** - maintain 85°F warm area
            3. **Hydration** - offer electrolyte solution
            4. **Quiet** - reduce noise and stress
            5. **Contact** an avian vet immediately
            
            *Technical Issue: {str(e)[:150]}*
            """)

st.caption("Educational tool only. Consult a veterinarian for medical emergencies.")
