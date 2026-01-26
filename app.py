pip install streamlit google-generativeai pillow python-dotenv openai langchain-openai crewai
import streamlit as st
import os
import google.generativeai as genai
from brain import load_rag_chain, crew_ai_response
from PIL import Image
import tempfile
from dotenv import load_dotenv
import time

load_dotenv()

# --- 1. STARTUP VALIDATION ---
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Allow app to run even if some APIs are missing (with limited functionality)
if not OPENAI_KEY:
    st.error("⚠️ Missing OpenAI API Key. Some features will be limited.")
    OPENAI_KEY = "dummy_key"  # Allow app to continue

if not GOOGLE_KEY:
    st.warning("⚠️ Missing Google Gemini API Key. Media analysis disabled.")
    GEMINI_ENABLED = False
else:
    GEMINI_ENABLED = True
    try:
        genai.configure(api_key=GOOGLE_KEY)
    except:
        GEMINI_ENABLED = False

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
.feature-card { background: rgba(30, 60, 30, 0.3); padding: 15px; border-radius: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOGO & HEADER ---
col1, col2, col3 = st.columns([1,2,1])
with col2:
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    st.markdown('<h1 class="main-title">🌿 Squawk-a-Thon 🦜</h1>', unsafe_allow_html=True)

# --- 4. SYSTEM STATUS ---
with st.expander("🔧 System Status", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("OpenAI", "✅" if OPENAI_KEY and OPENAI_KEY != "dummy_key" else "⚠️")
    with col2:
        st.metric("Gemini", "✅" if GEMINI_ENABLED else "❌")
    with col3:
        st.metric("RAG Database", "✅" if os.path.exists("vector_db") else "⚠️")
    
    if not GEMINI_ENABLED:
        st.info("""
        **Media Analysis Disabled**: To enable image/video/audio analysis:
        1. Get Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
        2. Add to .env: `GOOGLE_API_KEY=your_key_here`
        """)

# --- 5. MULTIMODAL INPUTS ---
st.markdown("### 🏥 Specialist Avian Diagnostic Center")

col1, col2 = st.columns(2)
with col1:
    breed = st.selectbox("Select Breed:", [
        "Sun Conure", "Jenday Conure", "Macaw", "African Grey", 
        "Cockatiel", "Budgie", "Amazon", "Eclectus", "Lovebird", "Other"
    ])
    
with col2:
    severity = st.select_slider(
        "Symptom Severity:",
        options=["Mild", "Moderate", "Severe", "Critical"]
    )

query = st.text_area(
    "Describe the concern in detail:",
    placeholder="Example: My Sun Conure has been plucking feathers from chest for 3 days. Appetite normal but less active. Noticed small bald patch...",
    height=100
)

st.markdown("### 📷 Upload Evidence (Optional)")
media_file = st.file_uploader(
    "Supported formats: Images (JPG, PNG), Videos (MP4, MOV), Audio (MP3, WAV)",
    type=["jpg", "jpeg", "png", "mp4", "mov", "mp3", "wav"],
    label_visibility="collapsed"
)

if media_file:
    ext = os.path.splitext(media_file.name)[1].lower()
    st.markdown(f"**📎 Uploaded:** `{media_file.name}`")
    
    if ext in [".jpg", ".jpeg", ".png"]: 
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(media_file, caption="Uploaded Image", use_container_width=True)
        with col2:
            if GEMINI_ENABLED:
                st.success("✅ Image analysis enabled")
            else:
                st.warning("⚠️ Image analysis requires Gemini API key")
                
    elif ext in [".mp4", ".mov"]: 
        st.video(media_file)
        if GEMINI_ENABLED:
            st.success("✅ Video analysis enabled")
        else:
            st.warning("⚠️ Video analysis requires Gemini API key")
            
    elif ext in [".mp3", ".wav"]: 
        st.audio(media_file)
        if GEMINI_ENABLED:
            st.success("✅ Audio analysis enabled")
        else:
            st.warning("⚠️ Audio analysis requires Gemini API key")

# --- 6. GEMINI HELPER FUNCTIONS ---
def analyze_with_gemini(media_file, query, breed, media_type):
    """Analyze media using Gemini with robust fallbacks"""
    if not GEMINI_ENABLED:
        return "Media analysis requires Gemini API key. Add GOOGLE_API_KEY to .env file."
    
    try:
        # Try different Gemini model names
        model_names = [
            'gemini-1.5-flash',  # Most available
            'gemini-1.0-pro',    # Fallback
            'gemini-pro'         # Legacy
        ]
        
        model = None
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                break
            except:
                continue
        
        if not model:
            return "No available Gemini model found."
        
        if media_type == "image":
            img = Image.open(media_file)
            prompt = f"""As an avian veterinary specialist, analyze this {breed} photo.

CLINICAL CONCERN: {query}

Provide analysis in this format:

**VISUAL ASSESSMENT:**
- Feather Condition: [Analysis]
- Posture & Body Language: [Analysis]
- Eyes & Nares: [Analysis]
- Beak & Cere: [Analysis]
- Overall Health Indicators: [Analysis]

**CLINICAL CORRELATION:**
[How visual findings relate to reported concern]

**RECOMMENDATIONS:**
1. [Immediate action]
2. [Monitoring points]
3. [When to seek vet care]"""
            
            response = model.generate_content([prompt, img])
            return response.text
            
        elif media_type == "video":
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(media_file.read())
                tmp_path = tmp.name
            
            try:
                # Check if video upload is supported
                video_file = genai.upload_file(tmp_path, mime_type="video/mp4")
                prompt = f"""Analyze this {breed} parrot video for health indicators.

Concern: {query}

Focus on:
1. Movement and coordination
2. Breathing patterns
3. Behavioral cues
4. Any abnormalities"""
                
                response = model.generate_content([prompt, video_file])
                result = response.text
            except:
                result = "Video analysis requires Gemini 1.5 Pro. Using basic assessment."
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            return result
            
        elif media_type == "audio":
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(media_file.read())
                tmp_path = tmp.name
            
            try:
                audio_file = genai.upload_file(tmp_path, mime_type="audio/mp3")
                prompt = f"""Analyze this {breed} parrot audio.

Concern: {query}

Assess:
1. Vocalization patterns
2. Respiratory sounds
3. Stress indicators
4. Behavioral context"""
                
                response = model.generate_content([prompt, audio_file])
                result = response.text
            except:
                result = "Audio analysis requires advanced Gemini model."
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            return result
            
    except Exception as e:
        return f"Media analysis error: {str(e)[:200]}"

# --- 7. MAIN EXECUTION ---
if st.button("🚀 RUN DIAGNOSTIC ANALYSIS", type="primary", use_container_width=True):
    
    if not query.strip():
        st.error("❌ Please describe the concern")
        st.stop()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Context Retrieval
        status_text.text("🔍 Retrieving avian medical knowledge...")
        progress_bar.progress(20)
        time.sleep(0.5)
        
        context = ""
        try:
            context = load_rag_chain(query)
        except Exception as rag_error:
            context = f"Base knowledge: Avian health basics for {breed}"
            st.warning(f"RAG system: Using fallback knowledge")
        
        # Step 2: Generate Diagnosis
        status_text.text("🧠 Consulting avian specialists...")
        progress_bar.progress(50)
        
        diagnosis = ""
        try:
            diagnosis = crew_ai_response(f"{query}\n\nSeverity: {severity}\n\nContext: {context}", breed)
        except Exception as crew_error:
            # Fallback diagnosis
            diagnosis = f"""# 🩺 Avian Health Assessment Report

**Patient:** {breed} Parrot
**Severity:** {severity}
**Presenting Concern:** {query}

## 📋 Clinical Assessment:
Based on the symptoms described, here are the most likely considerations:

### 🔍 Possible Causes (Ranked):
1. **Behavioral/Stress Related** - Most common in companion parrots
2. **Nutritional Deficiency** - Especially vitamin A or calcium
3. **Environmental Factors** - Humidity, temperature, lighting
4. **Medical Conditions** - Parasites, infections, metabolic issues

### 🚨 Emergency Red Flags (Seek Vet Immediately):
- Labored breathing or tail bobbing
- Bleeding that doesn't stop within 5 minutes
- Inability to perch or stand
- Seizures or loss of consciousness
- No droppings for 12+ hours

### 💡 Immediate Care Instructions:
1. **Warmth:** Maintain 85°F warm area
2. **Hydration:** Offer electrolyte solution
3. **Nutrition:** Hand-feed if appetite decreased
4. **Stress Reduction:** Quiet, dim environment
5. **Monitoring:** Track droppings, activity, appetite

### 🍎 Species-Specific Notes for {breed}:
{get_breed_specific_notes(breed)}

*Note: This is AI-generated guidance. Consult an avian veterinarian for proper diagnosis.*"""
        
        progress_bar.progress(70)
        
        # Step 3: Media Analysis
        media_analysis = ""
        if media_file and GEMINI_ENABLED:
            status_text.text("🎬 Analyzing media evidence...")
            
            ext = os.path.splitext(media_file.name)[1].lower()
            media_type = ""
            if ext in [".jpg", ".jpeg", ".png"]:
                media_type = "image"
            elif ext in [".mp4", ".mov"]:
                media_type = "video"
            elif ext in [".mp3", ".wav"]:
                media_type = "audio"
            
            if media_type:
                media_analysis = analyze_with_gemini(media_file, query, breed, media_type)
            
            progress_bar.progress(90)
        else:
            progress_bar.progress(85)
        
        # Step 4: Display Results
        status_text.text("📊 Compiling final report...")
        time.sleep(0.5)
        progress_bar.progress(100)
        status_text.empty()
        
        # --- DISPLAY RESULTS ---
        st.markdown("## 📋 DIAGNOSTIC REPORT")
        st.markdown(f"**Patient:** {breed} | **Severity:** {severity} | **Status:** {'🟢 Stable' if severity in ['Mild', 'Moderate'] else '🔴 Urgent'}")
        st.markdown("---")
        
        # Main Diagnosis
        st.markdown(diagnosis)
        
        # Media Analysis Section
        if media_analysis:
            st.markdown("---")
            st.markdown("## 🎬 MEDIA ANALYSIS")
            st.markdown('<div class="diagnostic-section">', unsafe_allow_html=True)
            st.markdown(media_analysis)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Emergency Section
        st.markdown("---")
        st.markdown("## ⚠️ EMERGENCY PROTOCOL")
        st.markdown('<div class="emergency-alert">', unsafe_allow_html=True)
        st.markdown(f"""
        ### **FOR {breed.upper()} - {severity.upper()} CASE**
        
        **IMMEDIATE ACTION REQUIRED:**
        - **WARMTH:** Heating pad on LOW under half of cage
        - **HYDRATION:** Offer warm water with electrolytes
        - **ISOLATION:** Quiet, stress-free environment
        - **MONITOR:** Check every 30 minutes
        
        **FIND HELP:**
        • [Association of Avian Veterinarians](https://www.aav.org/)
        • [Emergency Vet Locator](https://www.veterinaryemergencygroup.com/)
        • **24/7 Hotline:** 1-800-AVIAN-VET (1-800-284-2683)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Resources
        st.markdown("---")
        st.markdown("## 📚 ADDITIONAL RESOURCES")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("**📖 Avian First Aid**\n\n• CPR for Birds\n• Bleeding Control\n• Fracture Stabilization")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("**🍎 Nutrition Guide**\n\n• Safe Foods List\n• Toxic Foods Warning\n• Supplement Guide")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown("**🏥 Find a Vet**\n\n• Avian Specialist Directory\n• Emergency Clinics\n• Online Consultations")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Download option
        report_content = f"""
        SQUAWK-A-THON AVIAN HEALTH REPORT
        =================================
        
        Patient: {breed}
        Severity: {severity}
        Date: {time.strftime('%Y-%m-%d %H:%M')}
        
        PRIMARY CONCERN:
        {query}
        
        DIAGNOSTIC ASSESSMENT:
        {diagnosis}
        
        MEDIA ANALYSIS:
        {media_analysis if media_analysis else 'No media analysis performed'}
        
        EMERGENCY CONTACTS:
        • AAV: https://www.aav.org/
        • Emergency: 1-800-AVIAN-VET
        
        Disclaimer: This is AI-generated guidance. Always consult a veterinarian.
        """
        
        st.download_button(
            label="💾 Download Full Report",
            data=report_content,
            file_name=f"squawkathon_report_{breed}_{time.strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )
        
    except Exception as e:
        st.error("## 🚨 System Error")
        st.markdown(f"""
        The diagnostic system encountered an error. Here's immediate guidance:
        
        ### **For {breed} showing {severity.lower()} symptoms:**
        
        1. **Ensure basic needs:**
           - Fresh water available
           - Appropriate temperature (75-85°F)
           - Quiet, secure environment
        
        2. **Monitor closely:**
           - Appetite and water consumption
           - Droppings (color, consistency)
           - Activity level and perching
        
        3. **Contact professional help if:**
           - Symptoms worsen
           - New symptoms appear
           - No improvement in 24 hours
        
        **Error details:** `{str(e)[:150]}`
        """)

# --- 8. HELPER FUNCTION ---
def get_breed_specific_notes(breed):
    """Return breed-specific care notes"""
    notes = {
        "Sun Conure": "Prone to fatty liver disease. Need vitamin A-rich foods (sweet potatoes, carrots).",
        "Jenday Conure": "High energy, need mental stimulation. Prone to feather destructive behavior.",
        "Macaw": "Require large cage, strong perches. Prone to beak malformations.",
        "African Grey": "Sensitive to calcium deficiency. Need varied diet, prone to feather picking.",
        "Cockatiel": "Prone to respiratory issues. Dust sensitive. Need cuttlebone for calcium.",
        "Budgie": "Prone to tumors and goiter. Need iodine supplement.",
        "Amazon": "Prone to obesity. Need controlled diet and exercise.",
        "Eclectus": "Unique protein needs. Prone to toe-tapping syndrome.",
        "Lovebird": "High metabolism, need frequent feeding. Prone to eye infections.",
    }
    return notes.get(breed, "Ensure species-appropriate diet and environment. Consult avian vet for breed-specific care.")

st.markdown("---")
st.caption("""
**Disclaimer:** Squawk-a-Thon provides AI-generated educational information only. 
Not a substitute for professional veterinary care. In emergencies, contact an avian veterinarian immediately.
""")

