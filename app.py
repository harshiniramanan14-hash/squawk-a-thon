import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import tempfile
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize session state variables
if 'openai_available' not in st.session_state:
    st.session_state.openai_available = False
if 'gemini_available' not in st.session_state:
    st.session_state.gemini_available = False

# --- 1. INITIALIZATION ---
def initialize_ai_models():
    """Initialize AI models"""
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and len(openai_key) > 30:
        st.session_state.openai_available = True
    else:
        st.session_state.openai_available = False
    
    # Check Google Gemini API key
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key and len(google_key) > 30:
        try:
            genai.configure(api_key=google_key)
            # Try to create a simple model to test
            genai.GenerativeModel('gemini-pro')
            st.session_state.gemini_available = True
        except:
            st.session_state.gemini_available = False
    else:
        st.session_state.gemini_available = False

# Initialize models
initialize_ai_models()

# --- 2. UI SETUP ---
st.set_page_config(
    page_title="Squawk-a-Thon CrewAI 🦜", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                url('https://images.unsplash.com/photo-1552084117-56a987666449?q=80&w=2000');
    background-size: cover; 
    color: #f0fdf4;
}
.main-title { 
    color: #4ade80; 
    text-align: center; 
    font-size: 3.5rem; 
    text-shadow: 2px 2px #064e3b; 
    margin-top: -40px; 
}
.result-card { 
    background: rgba(255, 255, 255, 0.1); 
    backdrop-filter: blur(10px); 
    padding: 25px; 
    border-radius: 20px; 
    border: 1px solid #4ade80; 
    margin: 20px 0;
}
.agent-card { 
    background: rgba(30, 60, 90, 0.3); 
    padding: 15px; 
    border-radius: 10px; 
    margin: 10px 0; 
    border-left: 4px solid #4ade80;
}
.emergency-card { 
    background: rgba(255, 50, 50, 0.2); 
    padding: 15px; 
    border-radius: 10px; 
    border-left: 5px solid #ff4444;
    margin: 20px 0;
}
.sidebar-info {
    background: rgba(0, 50, 0, 0.3);
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# --- 3. HEADER ---
st.markdown('<h1 class="main-title">🦜 Squawk-a-Thon CrewAI</h1>', unsafe_allow_html=True)
st.markdown("### *Multi-Agent Avian Diagnostic System*")

# --- 4. AGENT INTRODUCTION ---
with st.expander("🤖 Meet Your Diagnostic Team", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        ### 👨‍⚕️ Chief Veterinarian
        **Dr. Aviana Feathers**
        • 30+ years experience
        • Board-certified avian vet
        • Differential diagnosis expert
        """)
    
    with col2:
        st.markdown("""
        ### 🧠 Behavior Specialist
        **Dr. Skye Talon**
        • Avian ethology PhD
        • Stress & behavior expert
        • Environmental enrichment
        """)
    
    with col3:
        st.markdown("""
        ### 🍎 Nutritionist
        **Dr. Peregrine Seed**
        • Avian nutrition specialist
        • Diet formulation expert
        • Supplementation guidance
        """)
    
    with col4:
        st.markdown("""
        ### 🚨 Emergency Coordinator
        **Dr. Falcon Emergency**
        • Critical care specialist
        • Triage and stabilization
        • Emergency protocols
        """)

# --- 5. MAIN INPUT SECTION ---
st.markdown("## 📝 Case Information")

# Breed selection
breed_options = [
    "Sun Conure", "African Grey", "Cockatiel", "Macaw", "Budgie", 
    "Amazon Parrot", "Eclectus", "Lovebird", "Quaker Parrot", "Other"
]

col1, col2 = st.columns(2)
with col1:
    breed = st.selectbox("Select Breed:", breed_options)
    if breed == "Other":
        breed = st.text_input("Specify breed:", placeholder="e.g., Senegal Parrot")

with col2:
    severity = st.select_slider(
        "Severity Level:",
        options=["Mild", "Moderate", "Severe", "Critical"],
        value="Moderate",
        help="Helps agents prioritize their analysis"
    )

# Symptom description
st.markdown("### 🩺 Describe Symptoms")
query = st.text_area(
    "Symptom Description:",
    height=150,
    placeholder="Example: My African Grey has been plucking chest feathers for 2 weeks. More noticeable in evenings. Appetite is normal but seems less vocal. No changes in droppings...",
    key="symptom_description"
)

# Media upload
st.markdown("### 📷 Supporting Evidence (Optional)")
media_file = st.file_uploader(
    "Upload photos, videos, or audio:",
    type=["jpg", "jpeg", "png", "mp4", "mov", "mp3", "wav"],
    help="Will be analyzed by the diagnostic team"
)

if media_file:
    ext = os.path.splitext(media_file.name)[1].lower()
    st.success(f"✅ Evidence uploaded: {media_file.name}")
    
    if ext in [".jpg", ".jpeg", ".png"]:
        st.image(media_file, caption="Uploaded Image", width=300)
    elif ext in [".mp4", ".mov"]:
        st.video(media_file)
    elif ext in [".mp3", ".wav"]:
        st.audio(media_file)

# --- 6. DIAGNOSTIC EXECUTION ---
st.markdown("---")

# Check system status
if not st.session_state.openai_available:
    st.error("⚠️ **System Alert:** OpenAI API key not found or invalid. Please add a valid OPENAI_API_KEY to your .env file.")
    st.info("Get your API key from: https://platform.openai.com/api-keys")

# Diagnostic button
if st.button("🚀 LAUNCH MULTI-AGENT DIAGNOSIS", type="primary", use_container_width=True):
    
    if not query or len(query.strip()) < 10:
        st.error("Please provide detailed symptom description (minimum 10 characters)")
        st.stop()
    
    if not st.session_state.openai_available:
        st.error("❌ CrewAI system requires OpenAI API key. Please configure your .env file.")
        st.stop()
    
    # Import here to avoid issues if APIs aren't available
    try:
        from brain import load_rag_chain, crew_ai_response
    except ImportError as e:
        st.error(f"❌ Failed to import brain module: {e}")
        st.stop()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("Initializing diagnostic protocol..."):
        
        try:
            # Step 1: RAG Context Retrieval
            status_text.text("🔍 Retrieving avian medical knowledge...")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            rag_context = load_rag_chain(query)
            
            # Step 2: CrewAI Analysis
            status_text.text("🤖 Assembling specialist team...")
            progress_bar.progress(30)
            
            # Show agent activation
            agent_cols = st.columns(4)
            agents = ["Chief Veterinarian", "Behavior Specialist", "Nutritionist", "Emergency Coordinator"]
            for idx, (col, agent) in enumerate(zip(agent_cols, agents)):
                with col:
                    st.success(f"✅ {agent}")
                progress_bar.progress(30 + (idx + 1) * 10)
                time.sleep(0.3)
            
            # Execute CrewAI
            status_text.text("🧠 Agents analyzing case collaboratively...")
            progress_bar.progress(70)
            
            diagnosis = crew_ai_response(f"{query} | Severity: {severity}", breed)
            
            # Step 3: Media Analysis (if available)
            media_analysis = None
            if media_file and st.session_state.gemini_available:
                status_text.text("🎬 Analyzing supporting evidence...")
                progress_bar.progress(85)
                
                try:
                    ext = os.path.splitext(media_file.name)[1].lower()
                    if ext in [".jpg", ".jpeg", ".png"]:
                        img = Image.open(media_file)
                        gemini_model = genai.GenerativeModel('gemini-pro')
                        media_prompt = f"""
                        As an avian diagnostic assistant, analyze this image of a {breed}.
                        
                        Case Context: {query}
                        Severity: {severity}
                        
                        Provide specific observations.
                        """
                        
                        response = gemini_model.generate_content([media_prompt, img])
                        media_analysis = response.text
                        
                except Exception as e:
                    media_analysis = f"Media analysis limited: {str(e)[:100]}"
            
            # Step 4: Final compilation
            status_text.text("📊 Compiling comprehensive report...")
            progress_bar.progress(95)
            time.sleep(0.5)
            progress_bar.progress(100)
            status_text.empty()
            
        except Exception as e:
            st.error(f"❌ Diagnostic process failed: {str(e)[:200]}")
            diagnosis = f"""
            # ⚠️ Diagnostic System Error
            
            An error occurred during the diagnostic process.
            
            **Error:** {str(e)[:150]}
            
            **Basic Guidance for {breed}:**
            - Monitor symptoms closely
            - Ensure proper temperature (75-85°F)
            - Provide fresh water and balanced diet
            - Contact avian veterinarian if symptoms persist
            
            **Emergency:** 1-800-AVIAN-VET
            """
            media_analysis = None
    
    # --- DISPLAY RESULTS ---
    st.markdown("## 📋 DIAGNOSTIC REPORT")
    
    # Case Header
    st.markdown(f"""
    <div class='agent-card'>
    <h3>🦜 Case Analysis: {breed}</h3>
    <p><strong>Severity:</strong> {severity} | <strong>Status:</strong> {'🟢 Stable' if severity in ['Mild', 'Moderate'] else '🔴 Urgent'}</p>
    <p><strong>Presenting Concern:</strong> {query[:100]}...</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Diagnosis
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(diagnosis)
    
    # Media Analysis
    if media_analysis:
        st.markdown("---")
        st.markdown("### 🎬 EVIDENCE ANALYSIS")
        st.markdown(f'<div class="agent-card">{media_analysis}</div>', unsafe_allow_html=True)
    
    # Action Plan
    st.markdown("---")
    st.markdown("### 🎯 INTEGRATED ACTION PLAN")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📋 Immediate Steps (Next 24h)
        1. Implement agent recommendations
        2. Monitor response to interventions
        3. Document observations hourly
        4. Prepare for vet consultation
        """)
    
    with col2:
        st.markdown("""
        #### 🔄 Follow-up Timeline
        - **4 hours:** Reassess critical parameters
        - **12 hours:** Evaluate intervention effectiveness
        - **24 hours:** Decide on veterinary consultation
        - **48 hours:** Full case review
        """)
    
    # Emergency Section
    if severity in ["Severe", "Critical"]:
        st.markdown("---")
        st.markdown("### ⚠️ EMERGENCY PROTOCOL")
        st.markdown('<div class="emergency-card">', unsafe_allow_html=True)
        st.markdown(f"""
        ## 🚨 URGENT ACTION REQUIRED - {severity.upper()} CASE
        
        **Immediate Stabilization:**
        1. **Isolate** in quiet, warm (85°F) area
        2. **Hydrate** with electrolyte solution
        3. **Monitor** breathing, consciousness, position
        4. **Prepare** for emergency transport
        
        **Contact Resources:**
        - **Emergency Hotline:** 1-800-AVIAN-VET
        - **Poison Control:** (888) 426-4435
        - **Nearest Avian ER:** [Find at aav.org](https://www.aav.org/)
        
        **Do NOT delay veterinary care for severe/critical cases.**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Download Report
    report_content = f"""
    SQUAWK-A-THON CREWAI DIAGNOSTIC REPORT
    ======================================
    Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
    Breed: {breed}
    Severity: {severity}
    
    SYMPTOM DESCRIPTION:
    {query}
    
    DIAGNOSTIC ANALYSIS:
    {diagnosis}
    
    MEDIA ANALYSIS:
    {media_analysis if media_analysis else 'No media analysis performed'}
    
    ACTION PLAN:
    {'URGENT VETERINARY ATTENTION REQUIRED' if severity in ['Severe', 'Critical'] else 'Monitor and implement recommendations'}
    
    DISCLAIMER:
    This AI-generated report is for informational purposes only.
    Always consult with a certified avian veterinarian.
    """
    
    st.download_button(
        "📥 Download Complete Report",
        data=report_content,
        file_name=f"squawkathon_{breed.replace(' ', '_')}_{time.strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- 7. SIDEBAR ---
with st.sidebar:
    st.markdown("## 🔧 System Status")
    
    if st.session_state.openai_available:
        st.success("✅ CrewAI System: Operational")
    else:
        st.error("❌ CrewAI System: API Key Missing")
        st.info("Add OPENAI_API_KEY to .env file")
    
    if st.session_state.gemini_available:
        st.success("✅ Media Analysis: Available")
    else:
        st.warning("⚠️ Media Analysis: Limited")
    
    st.markdown("---")
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.markdown("""
    ## 🦜 About This System
    
    **Squawk-a-Thon CrewAI** uses specialized agents:
    
    1. **Chief Veterinarian** - Medical diagnosis
    2. **Behavior Specialist** - Stress & behavior
    3. **Nutritionist** - Diet & supplements
    4. **Emergency Coordinator** - Urgency assessment
    
    Each query gets a unique, tailored response.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 📚 Resources")
    st.markdown("""
    - [Association of Avian Veterinarians](https://www.aav.org/)
    - [Avian First Aid Guide](https://www.aav.org/page/FirstAid)
    - [Parrot Nutrition](https://lafeber.com/pet-birds/)
    - **Emergency:** 1-800-AVIAN-VET
    """)
    
    st.markdown("---")
    st.markdown("## ⚠️ Important Notice")
    st.markdown("""
    This is an AI-powered educational tool.
    
    **NOT** a substitute for veterinary care.
    
    Always consult a certified avian veterinarian for medical concerns.
    """)

# --- 8. FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #aaa; font-size: 0.9em; padding: 20px;'>
<p><strong>🦜 Squawk-a-Thon CrewAI Diagnostic System</strong></p>
<p>Powered by CrewAI multi-agent technology</p>
<p><em>For educational purposes only. Consult avian veterinarian for medical care.</em></p>
</div>
""", unsafe_allow_html=True)
