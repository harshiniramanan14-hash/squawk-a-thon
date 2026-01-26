import streamlit as st
import os
import google.generativeai as genai
from brain import load_rag_chain, crew_ai_response
from PIL import Image
import tempfile
from dotenv import load_dotenv
import base64

load_dotenv()

# --- 1. STARTUP VALIDATION ---
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if not GOOGLE_KEY or not OPENAI_KEY:
    st.error("🚨 Missing API Keys! Ensure GOOGLE_API_KEY and OPENAI_API_KEY are in your .env file.")
    st.stop()

# Configure Gemini with correct API version
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
.diagnostic-section { background: rgba(0, 100, 0, 0.2); padding: 15px; border-radius: 10px; margin: 10px 0; }
.emergency-alert { background: rgba(255, 0, 0, 0.2); padding: 15px; border-radius: 10px; border-left: 5px solid red; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGO & HEADER ---
col1, col2, col3 = st.columns([1,2,1])
with col2:
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    st.markdown('<h1 class="main-title">🌿 Squawk-a-Thon 🦜</h1>', unsafe_allow_html=True)

# --- 4. MULTIMODAL INPUTS ---
st.write("### 🏥 Specialist Avian Diagnostic Center")
breed = st.selectbox("Select Breed:", ["Sun Conure", "Jenday Conure", "Macaw", "African Grey", "Cockatiel", "Budgie", "Other (Specify)"])
if breed == "Other (Specify)":
    breed = st.text_input("Enter breed name:")

query = st.text_area("Describe the concern (e.g., feather plucking, lethargy, wing drooping):", 
                    placeholder="Be specific: duration, severity, recent changes...")

media_file = st.file_uploader("📷 Upload Evidence (Photo/Video/Audio)", 
                             type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mp3", "wav", "m4a"])

if media_file:
    ext = os.path.splitext(media_file.name)[1].lower()
    st.write(f"**Uploaded:** {media_file.name}")
    
    if ext in [".jpg", ".jpeg", ".png"]: 
        st.image(media_file, width=400, caption="Uploaded Image")
    elif ext in [".mp4", ".mov", ".avi"]: 
        st.video(media_file)
        st.info("🎬 Video analysis enabled - analyzing frames for behavioral cues")
    elif ext in [".mp3", ".wav", ".m4a"]: 
        st.audio(media_file)
        st.info("🎵 Audio analysis enabled - analyzing vocalizations and sounds")

# --- 5. GEMINI MODEL SELECTION ---
@st.cache_resource
def get_gemini_model():
    """Get the best available Gemini model"""
    try:
        # Try Gemini 1.5 Pro (supports multimodal including video/audio)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        st.success("✅ Using Gemini 1.5 Pro (Multimodal)")
        return model
    except Exception as e:
        try:
            # Fallback to Gemini Pro Vision for images
            model = genai.GenerativeModel('gemini-pro-vision')
            st.info("⚠️ Using Gemini Pro Vision (Images only)")
            return model
        except Exception as e2:
            st.error(f"❌ Gemini API Error: {e2}")
            return None

# --- 6. EXECUTION ---
if st.button("🚀 RUN MULTIMODAL DIAGNOSTIC", type="primary", use_container_width=True):
    
    # Validate inputs
    if not query.strip():
        st.error("Please describe the concern")
        st.stop()
    
    with st.spinner("🦜 Consulting avian specialists across the rainforest..."):
        try:
            # Get Gemini model
            gemini_model = get_gemini_model()
            if not gemini_model:
                st.error("Gemini API unavailable. Using text-only analysis.")
            
            # 1. RAG Context Retrieval
            with st.expander("🔍 Retrieving Medical Context", expanded=False):
                context = load_rag_chain(query)
                if context:
                    st.info(f"Retrieved {len(context.split())} words of avian medical context")
            
            # 2. CrewAI Diagnostic Report
            with st.expander("🧠 Generating Expert Diagnosis", expanded=False):
                try:
                    report = crew_ai_response(f"{query}\n\nContext: {context}", breed)
                except Exception as crew_error:
                    st.warning(f"CrewAI system issue: {str(crew_error)[:100]}")
                    # Fallback to direct Gemini diagnosis
                    if gemini_model:
                        fallback_prompt = f"""You are Dr. Aviana Feathers, a senior avian veterinarian with 25 years of experience.

PATIENT: {breed} parrot
PRESENTING CONCERN: {query}
MEDICAL CONTEXT: {context}

Please provide a comprehensive veterinary assessment including:

1. **Differential Diagnosis** (list 3-5 most likely causes ranked by probability)
2. **Clinical Signs Analysis** (what each symptom might indicate)
3. **Immediate Care Instructions** (what to do in next 24 hours)
4. **Dietary Recommendations** (specific to {breed})
5. **Environmental Adjustments**
6. **Red Flags** (when to seek emergency vet care)
7. **Follow-up Monitoring** (what to watch for)

Format with clear headings and bullet points. Be specific and actionable."""
                        
                        report = gemini_model.generate_content(fallback_prompt).text
                    else:
                        report = f"""# 🩺 Avian Health Assessment for {breed}

## Primary Concern: {query}

### Immediate Actions:
1. **Temperature Control** - Maintain 85°F warm area
2. **Hydration** - Offer electrolyte solution (Pedialyte)
3. **Nutrition** - Hand-feed if not eating
4. **Stress Reduction** - Quiet, dim environment
5. **Monitor** - Droppings, breathing, activity level

### Emergency Signs (Vet Immediately):
• Labored breathing or tail bobbing
• Bleeding that doesn't stop
• Inability to perch
• Seizures or tremors
• No droppings for 12+ hours

*System temporarily limited. Consult certified avian veterinarian.*"""
            
            # 3. Multimodal Analysis (Images/Video/Audio)
            vision_out = ""
            if media_file and gemini_model:
                with st.expander("🎬 Analyzing Media Evidence", expanded=False):
                    try:
                        if ext in [".jpg", ".jpeg", ".png"]:
                            # IMAGE ANALYSIS
                            img = Image.open(media_file)
                            img_prompt = f"""As an avian veterinary imaging specialist, analyze this {breed} parrot photo.

CLINICAL CONTEXT: {query}

Analyze for:
1. **Feather Condition** - plucking, barbering, discoloration
2. **Posture & Stance** - wing position, weight bearing
3. **Eyes & Nares** - discharge, swelling, symmetry
4. **Beak & Cere** - overgrowth, discoloration
5. **General Condition** - body score, alertness
6. **Environmental Clues** - perches, cage conditions

Provide specific observations and clinical correlation."""
                            
                            vision_out = gemini_model.generate_content([img_prompt, img]).text
                            
                        elif ext in [".mp4", ".mov", ".avi"]:
                            # VIDEO ANALYSIS
                            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                                tmp.write(media_file.read())
                                tmp_path = tmp.name
                            
                            # Upload video to Gemini
                            video_file = genai.upload_file(tmp_path)
                            
                            video_prompt = f"""As an avian behavior specialist, analyze this {breed} parrot video.

CLINICAL CONCERN: {query}

Analyze for:
1. **Locomotion** - gait abnormalities, wing use
2. **Behavior** - activity level, interaction
3. **Postural Changes** - fluffing, trembling, leaning
4. **Respiratory Signs** - tail bobbing, open-mouth breathing
5. **Vocalizations** - changes in frequency/pitch
6. **Neurological Signs** - head tilt, circling, seizures

Timestamp key observations and rate severity (mild/moderate/severe)."""
                            
                            vision_out = gemini_model.generate_content([video_prompt, video_file]).text
                            os.unlink(tmp_path)
                            
                        elif ext in [".mp3", ".wav", ".m4a"]:
                            # AUDIO ANALYSIS
                            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                                tmp.write(media_file.read())
                                tmp_path = tmp.name
                            
                            audio_file = genai.upload_file(tmp_path)
                            
                            audio_prompt = f"""As an avian acoustic specialist, analyze this {breed} parrot audio.

CLINICAL CONCERN: {query}

Analyze for:
1. **Vocalization Patterns** - changes in frequency, duration
2. **Respiratory Sounds** - wheezing, clicking, sneezing
3. **Distress Calls** - frequency and intensity
4. **Behavioral Context** - what sounds might indicate
5. **Comparative Analysis** - vs. normal {breed} vocalizations

Rate abnormality level and suggest possible causes."""
                            
                            vision_out = gemini_model.generate_content([audio_prompt, audio_file]).text
                            os.unlink(tmp_path)
                            
                    except Exception as vision_error:
                        vision_out = f"## Media Analysis Limited\n\n*Technical note: {str(vision_error)[:200]}*\n\nFocus on the clinical assessment above."
            
            # --- DISPLAY RESULTS ---
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            # Main Report
            st.markdown("## 📋 Comprehensive Diagnostic Report")
            st.markdown(f"**Patient:** {breed} | **Presenting Concern:** {query}")
            st.markdown("---")
            
            # Display Report
            st.markdown(report)
            
            # Media Analysis Section
            if vision_out:
                st.markdown("---")
                st.markdown("## 🎬 Media Analysis Findings")
                st.markdown('<div class="diagnostic-section">', unsafe_allow_html=True)
                st.markdown(vision_out)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Emergency Guidance
            st.markdown("---")
            st.markdown("## ⚠️ Emergency Action Card")
            st.markdown('<div class="emergency-alert">', unsafe_allow_html=True)
            st.markdown("""
            ### **IMMEDIATE VETERINARY CARE REQUIRED IF:**
            • **Respiratory Distress** (open-mouth breathing, tail bobbing)  
            • **Trauma** (bleeding, fractures, wounds)  
            • **Neurological Signs** (seizures, circling, inability to stand)  
            • **Toxic Exposure** (known ingestion of toxins)  
            • **Prolapsed Organ** (tissue protruding from vent)  
            
            **24/7 Avian Emergency Hotline: 1-800-AVIAN-VET**
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download Report
            report_text = f"Squawk-a-Thon Report\nBreed: {breed}\nConcern: {query}\n\n{report}\n\nMedia Analysis:\n{vision_out}"
            st.download_button("💾 Download Full Report", report_text, file_name=f"squawkathon_report_{breed}.txt")
            
        except Exception as e:
            st.error("## 🚨 Diagnostic System Encountered an Issue")
            st.markdown('<div class="emergency-alert">', unsafe_allow_html=True)
            st.markdown(f"""
            ### **For {breed} with {query}:**
            
            **CRITICAL FIRST AID:**
            1. **WARMTH** - Provide 85-90°F heat source (heating pad on LOW under half the cage)
            2. **HYDRATION** - Offer warm electrolyte solution via syringe if not drinking
            3. **ISOLATION** - Quiet, stress-free environment away from other pets
            4. **MONITOR** - Check droppings, breathing rate, and consciousness every 30 minutes
            
            **FIND AN AVIAN VET:**
            • [Association of Avian Veterinarians](https://www.aav.org/)
            • [Avian Vet Locator](https://abvp.com/animal-owners/find-a-specialist/)
            
            *Technical Error: {str(e)[:200]}*
            """)
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("""
**Disclaimer:** Squawk-a-Thon is an AI-powered educational tool for informational purposes only. 
It is not a substitute for professional veterinary care. Always consult with a certified avian veterinarian for medical concerns.
""")
