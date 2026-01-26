import streamlit as st
import os
import google.generativeai as genai
from openai import OpenAI
from PIL import Image
import tempfile
from dotenv import load_dotenv
import time
import json
from typing import Dict, List, Optional

load_dotenv()

# --- 1. INITIALIZE AI MODELS ---
@st.cache_resource
def initialize_models():
    """Initialize AI models with proper configuration"""
    models = {}
    
    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            models['openai'] = OpenAI(api_key=openai_key)
            st.session_state.openai_available = True
        except:
            st.session_state.openai_available = False
    else:
        st.session_state.openai_available = False
    
    # Gemini
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        try:
            genai.configure(api_key=google_key)
            # Try different model names
            try:
                models['gemini'] = genai.GenerativeModel('gemini-1.5-pro')
            except:
                try:
                    models['gemini'] = genai.GenerativeModel('gemini-pro')
                except:
                    models['gemini'] = None
            st.session_state.gemini_available = models['gemini'] is not None
        except:
            st.session_state.gemini_available = False
    else:
        st.session_state.gemini_available = False
    
    return models

models = initialize_models()

# --- 2. VETERINARY KNOWLEDGE BASE ---
AVIAN_MEDICAL_KNOWLEDGE = {
    "feather_issues": {
        "plucking": {
            "causes": ["Stress/anxiety", "Boredom", "Nutritional deficiencies", "Parasites", "Allergies", "Hormonal imbalances"],
            "diagnostics": ["Skin scrape", "Blood work", "Fecal exam", "Environmental assessment"],
            "treatments": ["Environmental enrichment", "Diet improvement", "Behavior modification", "Medical treatment if needed"]
        },
        "loss": {
            "causes": ["Molting", "Stress", "Infection", "Nutritional issues", "Autoimmune disorders"],
            "diagnostics": ["Physical exam", "Blood tests", "Skin biopsy"],
            "treatments": ["Supportive care", "Address underlying cause", "Nutritional support"]
        }
    },
    "respiratory": {
        "sneezing": {
            "causes": ["Dust/allergens", "Respiratory infection", "Sinusitis", "Foreign body"],
            "diagnostics": ["Physical exam", "X-rays", "Culture/sensitivity"],
            "treatments": ["Environmental cleanup", "Antibiotics if bacterial", "Supportive care"]
        },
        "wheezing": {
            "causes": ["Air sac infection", "Pneumonia", "Heart disease", "Allergies"],
            "diagnostics": ["X-rays", "Blood work", "Tracheal wash"],
            "treatments": ["Antibiotics", "Anti-inflammatories", "Environmental management"]
        }
    },
    "gastrointestinal": {
        "diarrhea": {
            "causes": ["Dietary indiscretion", "Infection", "Parasites", "Liver disease", "Toxicity"],
            "diagnostics": ["Fecal exam", "Blood work", "X-rays", "Culture"],
            "treatments": ["Fluid therapy", "Diet modification", "Medications as needed"]
        },
        "vomiting": {
            "causes": ["Infection", "Toxicity", "Metabolic disorder", "Gastrointestinal obstruction"],
            "diagnostics": ["Physical exam", "X-rays", "Blood work"],
            "treatments": ["Supportive care", "Address underlying cause", "Medications"]
        }
    },
    "neurological": {
        "seizures": {
            "causes": ["Toxicity", "Infection", "Metabolic disorder", "Trauma", "Neoplasia"],
            "diagnostics": ["Blood work", "X-rays", "Neurological exam"],
            "treatments": ["Emergency stabilization", "Anti-seizure medications", "Address underlying cause"]
        },
        "head_tilt": {
            "causes": ["Ear infection", "Neurological disorder", "Trauma", "Toxicity"],
            "diagnostics": ["Physical exam", "X-rays", "Blood work", "Neurological exam"],
            "treatments": ["Antibiotics if infection", "Supportive care", "Address underlying cause"]
        }
    }
}

# --- 3. DIAGNOSTIC AI FUNCTIONS ---
def analyze_symptoms_with_ai(breed: str, query: str, severity: str) -> str:
    """Use AI to analyze symptoms and generate personalized diagnosis"""
    
    # Prepare context from knowledge base
    context = extract_relevant_knowledge(query)
    
    # Generate AI response
    return generate_ai_diagnosis(breed, query, severity, context)

def extract_relevant_knowledge(query: str) -> Dict:
    """Extract relevant medical knowledge for the query"""
    query_lower = query.lower()
    relevant_info = {}
    
    for category, symptoms in AVIAN_MEDICAL_KNOWLEDGE.items():
        for symptom, info in symptoms.items():
            if symptom in query_lower or any(word in query_lower for word in symptom.split('_')):
                relevant_info[symptom] = info
    
    return relevant_info

def generate_ai_diagnosis(breed: str, query: str, severity: str, context: Dict) -> str:
    """Generate diagnosis using OpenAI GPT"""
    
    if not st.session_state.openai_available:
        return generate_fallback_diagnosis(breed, query, severity, context)
    
    try:
        # Create detailed prompt based on query
        prompt = create_diagnostic_prompt(breed, query, severity, context)
        
        response = models['openai'].chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": """You are Dr. Aviana Feathers, a board-certified avian veterinarian with 25 years of experience.
                    You are analyzing a parrot patient and need to provide a comprehensive diagnostic assessment.
                    
                    CRITICAL: Your response MUST be tailored to the SPECIFIC SYMPTOMS described.
                    DO NOT give generic advice. ANALYZE EACH SYMPTOM mentioned.
                    
                    Format your response exactly as follows:
                    
                    # 🩺 DIAGNOSTIC ASSESSMENT
                    
                    ## 📋 PRESENTING CASE
                    [Brief summary specific to this patient]
                    
                    ## 🔍 SYMPTOM-BY-SYMPTOM ANALYSIS
                    [For EACH symptom mentioned, provide specific analysis]
                    
                    ## 🧪 DIFFERENTIAL DIAGNOSIS (Ranked)
                    1. [Most likely - with specific reasoning]
                    2. [Second likely - with specific reasoning]
                    3. [Third possibility - with specific reasoning]
                    
                    ## 🚨 URGENCY ASSESSMENT
                    [Based on severity: Mild/Moderate/Severe/Critical - with specific justification]
                    
                    ## 💊 IMMEDIATE RECOMMENDATIONS
                    [Specific actions for THIS case]
                    
                    ## 📊 MONITORING PARAMETERS
                    [What to watch for specific to these symptoms]
                    
                    ## ⚠️ EMERGENCY RED FLAGS
                    [Specific worsening signs for THESE symptoms]
                    """
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"AI Analysis Error: {str(e)[:200]}")
        return generate_fallback_diagnosis(breed, query, severity, context)

def create_diagnostic_prompt(breed: str, query: str, severity: str, context: Dict) -> str:
    """Create detailed prompt for AI analysis"""
    
    # Extract key symptoms from query
    symptoms = []
    for category, symptom_dict in AVIAN_MEDICAL_KNOWLEDGE.items():
        for symptom in symptom_dict.keys():
            if symptom.replace('_', ' ') in query.lower():
                symptoms.append(symptom.replace('_', ' '))
    
    if not symptoms:
        # Look for symptom-related keywords
        symptom_keywords = ['pluck', 'feather', 'sneeze', 'cough', 'diarrhea', 'vomit', 'letharg', 'sleep', 'eat', 'drink', 'breath', 'wheeze']
        for keyword in symptom_keywords:
            if keyword in query.lower():
                symptoms.append(keyword)
    
    return f"""
    PATIENT CONSULTATION REQUEST
    
    BREED: {breed}
    SEVERITY LEVEL: {severity}
    
    OWNER'S DESCRIPTION:
    "{query}"
    
    IDENTIFIED SYMPTOMS: {', '.join(symptoms) if symptoms else 'Various symptoms described'}
    
    RELEVANT MEDICAL CONTEXT:
    {json.dumps(context, indent=2) if context else 'No specific medical patterns identified'}
    
    ADDITIONAL DETAILS:
    - Current time: {time.strftime('%Y-%m-%d %H:%M')}
    - Query length: {len(query)} characters
    - Symptom keywords found: {len(symptoms)}
    
    Please analyze this SPECIFIC case and provide a tailored diagnostic assessment.
    Focus on the exact symptoms described by the owner.
    """

def generate_fallback_diagnosis(breed: str, query: str, severity: str, context: Dict) -> str:
    """Generate a detailed fallback diagnosis when AI is unavailable"""
    
    # Analyze query for key information
    query_lower = query.lower()
    
    # Identify potential issues
    issues = []
    if 'pluck' in query_lower or 'feather' in query_lower:
        issues.append("Feather destructive behavior")
    if 'sneeze' in query_lower or 'cough' in query_lower:
        issues.append("Respiratory concerns")
    if 'diarrhea' in query_lower or 'loose' in query_lower:
        issues.append("Gastrointestinal issues")
    if 'letharg' in query_lower or 'tired' in query_lower:
        issues.append("Lethargy/Decreased activity")
    if 'not eat' in query_lower or 'appetite' in query_lower:
        issues.append("Appetite changes")
    if 'breath' in query_lower or 'wheeze' in query_lower:
        issues.append("Breathing abnormalities")
    
    if not issues:
        issues = ["General health concern"]
    
    return f"""
# 🩺 DIAGNOSTIC ASSESSMENT - AI ANALYSIS

## 📋 PRESENTING CASE
{breed} parrot presenting with {severity.lower()} symptoms as described: "{query[:200]}..."

## 🔍 SYMPTOM ANALYSIS
Based on your description, the following concerns are noted:
{chr(10).join(f"• **{issue}** - Requires further evaluation" for issue in issues)}

## 🧪 DIFFERENTIAL CONSIDERATIONS
1. **Behavioral/Environmental Factors** - Common in companion birds experiencing stress or environmental changes
2. **Nutritional Imbalance** - Many avian health issues relate to diet deficiencies or excesses
3. **Early Disease Process** - Could indicate developing medical condition needing professional assessment

## 🚨 URGENCY ASSESSMENT
**Severity: {severity}**
- {'Low immediate risk but monitoring needed' if severity in ['Mild', 'Moderate'] else 'Higher concern requiring prompt attention'}

## 💊 IMMEDIATE RECOMMENDATIONS
1. **Environmental Optimization**
   - Temperature: 75-85°F
   - Humidity: 40-60%
   - Quiet, stress-free location

2. **Supportive Care**
   - Fresh water always available
   - Offer favorite foods to encourage eating
   - Monitor droppings closely

3. **Observation Protocol**
   - Record symptoms every 2-4 hours
   - Note any changes in behavior
   - Track food/water consumption

## 📊 SPECIFIC MONITORING FOR THIS CASE
- **Appetite:** Should eat within 4-6 hours of waking
- **Droppings:** Monitor color, consistency, frequency
- **Activity:** Note energy levels throughout day
- **Breathing:** Watch for any abnormalities

## ⚠️ EMERGENCY INDICATORS
Seek immediate veterinary care if:
• Labored breathing or tail bobbing develops
• Inability to perch or maintain balance
• Bleeding that doesn't stop within 5 minutes
• Seizures or loss of consciousness
• Significant worsening of described symptoms

*Note: This is general guidance. Specific symptoms require specific analysis.*
"""

def analyze_media_with_ai(media_file, breed: str, query: str) -> Optional[str]:
    """Analyze media files using AI"""
    if not st.session_state.gemini_available or not media_file:
        return None
    
    try:
        ext = os.path.splitext(media_file.name)[1].lower()
        
        # Create query-specific prompt
        media_prompt = f"""
        Analyze this media of a {breed} parrot.
        
        Owner's concern: {query}
        
        Please provide specific observations about:
        1. Visible physical signs related to the concern
        2. Behavioral cues evident in the media
        3. Environmental factors visible
        4. Any abnormalities noted
        5. Recommendations based on visual/audio evidence
        
        Be specific and relate findings directly to: {query}
        """
        
        if ext in [".jpg", ".jpeg", ".png"]:
            img = Image.open(media_file)
            response = models['gemini'].generate_content([media_prompt, img])
            return response.text
            
        elif ext in [".mp4", ".mov"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(media_file.read())
                tmp_path = tmp.name
            
            try:
                video_file = genai.upload_file(tmp_path)
                response = models['gemini'].generate_content([media_prompt, video_file])
                result = response.text
            except:
                result = "Video analysis requires advanced features. Focus on: mobility, coordination, activity level."
            finally:
                os.unlink(tmp_path)
            return result
            
        elif ext in [".mp3", ".wav"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(media_file.read())
                tmp_path = tmp.name
            
            try:
                audio_file = genai.upload_file(tmp_path)
                response = models['gemini'].generate_content([media_prompt, audio_file])
                result = response.text
            except:
                result = "Audio analysis requires advanced features. Listen for: changes in vocalization patterns."
            finally:
                os.unlink(tmp_path)
            return result
            
    except Exception as e:
        return f"Media analysis encountered limitations: {str(e)[:150]}"
    
    return None

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Squawk-a-Thon AI 🦜", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                url('https://images.unsplash.com/photo-1552084117-56a987666449?q=80&w=2000');
    background-size: cover; color: #f0fdf4;
}
.main-title { color: #4ade80; text-align: center; font-size: 3.5rem; text-shadow: 2px 2px #064e3b; margin-top: -40px; }
.result-card { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); padding: 25px; border-radius: 20px; border: 1px solid #4ade80; }
.symptom-highlight { background: rgba(255, 215, 0, 0.2); padding: 10px; border-radius: 8px; margin: 5px 0; }
.breed-card { background: rgba(139, 69, 19, 0.3); padding: 15px; border-radius: 10px; }
.emergency-box { background: rgba(255, 0, 0, 0.15); padding: 15px; border-radius: 10px; border-left: 5px solid #ff4444; }
</style>
""", unsafe_allow_html=True)

# --- 5. HEADER ---
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    st.markdown('<h1 class="main-title">🦜 Squawk-a-Thon AI Diagnostic</h1>', unsafe_allow_html=True)
    st.markdown("### *Intelligent Avian Health Analysis*")

# --- 6. SYSTEM STATUS ---
with st.expander("🔍 AI System Status", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "✅ Operational" if st.session_state.openai_available else "⚠️ Limited"
        st.metric("Diagnostic AI", status)
    with col2:
        status = "✅ Available" if st.session_state.gemini_available else "❌ Unavailable"
        st.metric("Media Analysis", status)
    with col3:
        st.metric("Response Quality", "Tailored per Query")

# --- 7. INPUT SECTION ---
st.markdown("## 🏥 Enter Case Details")

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    breed_options = [
        "Sun Conure", "Jenday Conure", "Macaw", "African Grey", 
        "Cockatiel", "Budgie", "Amazon Parrot", "Eclectus Parrot",
        "Lovebird", "Quaker Parrot", "Other"
    ]
    breed = st.selectbox("Select Bird Breed:", breed_options)
    
    if breed == "Other":
        breed = st.text_input("Specify breed:", placeholder="e.g., Senegal Parrot")

with col2:
    severity = st.select_slider(
        "Assess Severity Level:",
        options=["Mild", "Moderate", "Severe", "Critical"],
        value="Moderate",
        help="Mild: Minor concern | Critical: Emergency situation"
    )

# Symptom description with guidance
st.markdown("### 📝 Describe Symptoms in Detail")
st.info("""
**Be specific for better analysis:** Include duration, frequency, changes noticed, and any triggers.
Example: _"My Sun Conure has been plucking chest feathers for 3 days, mainly in the evening. 
Appetite is normal but he's sleeping more. No changes in droppings. Recently moved cage location."_
""")

query = st.text_area(
    "Symptom Description:",
    height=150,
    placeholder="Describe all symptoms, duration, and any relevant details...",
    key="symptom_input"
)

# Media upload
st.markdown("### 📷 Upload Supporting Media (Optional)")
media_file = st.file_uploader(
    "Add photos, videos, or audio recordings:",
    type=["jpg", "jpeg", "png", "mp4", "mov", "mp3", "wav"],
    help="Helpful for: physical signs (photos), behavior (videos), sounds (audio)"
)

if media_file:
    ext = os.path.splitext(media_file.name)[1].lower()
    st.success(f"✅ Media uploaded: {media_file.name}")
    
    if ext in [".jpg", ".jpeg", ".png"]:
        st.image(media_file, caption="Uploaded Image", width=300)
    elif ext in [".mp4", ".mov"]:
        st.video(media_file)
    elif ext in [".mp3", ".wav"]:
        st.audio(media_file)

# --- 8. ANALYSIS BUTTON ---
st.markdown("---")
if st.button("🤖 GENERATE AI DIAGNOSIS", type="primary", use_container_width=True):
    
    if not query or len(query.strip()) < 10:
        st.error("Please provide a detailed symptom description (at least 10 characters)")
        st.stop()
    
    # Show analysis progress
    with st.spinner("🔍 AI is analyzing your specific case..."):
        progress_bar = st.progress(0)
        
        # Step 1: Initial analysis
        status_text = st.empty()
        status_text.text("Analyzing symptom patterns...")
        progress_bar.progress(25)
        time.sleep(0.5)
        
        # Step 2: Generate diagnosis
        status_text.text("Consulting avian medical knowledge...")
        diagnosis = analyze_symptoms_with_ai(breed, query, severity)
        progress_bar.progress(60)
        
        # Step 3: Media analysis
        media_analysis = None
        if media_file:
            status_text.text("Analyzing media evidence...")
            media_analysis = analyze_media_with_ai(media_file, breed, query)
            progress_bar.progress(85)
        else:
            progress_bar.progress(80)
        
        # Step 4: Compile results
        status_text.text("Compiling comprehensive report...")
        time.sleep(0.5)
        progress_bar.progress(100)
        status_text.empty()
    
    # --- DISPLAY RESULTS ---
    st.markdown("## 📊 AI DIAGNOSTIC REPORT")
    
    # Query summary
    st.markdown(f"""
    <div class='breed-card'>
    <strong>Case Summary:</strong> {breed} | <strong>Severity:</strong> {severity}<br>
    <strong>Presenting Concern:</strong> {query[:150]}...
    </div>
    """, unsafe_allow_html=True)
    
    # Main Diagnosis
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(diagnosis)
    
    # Media Analysis if available
    if media_analysis:
        st.markdown("---")
        st.markdown("### 🎬 MEDIA ANALYSIS")
        st.markdown('<div class="symptom-highlight">', unsafe_allow_html=True)
        st.markdown(media_analysis)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Emergency Section
    st.markdown("---")
    st.markdown("### ⚠️ ACTION PLAN")
    st.markdown('<div class="emergency-box">', unsafe_allow_html=True)
    
    if severity in ["Severe", "Critical"]:
        st.markdown("""
        ## 🚨 IMMEDIATE ACTION REQUIRED
        
        **Based on severity assessment, this case requires prompt veterinary attention.**
        
        **Next Steps:**
        1. **Contact avian veterinarian immediately**
        2. **Prepare for vet visit:**
           - Bring recent droppings sample
           - Note all symptoms and timeline
           - Bring current diet sample
        3. **Emergency stabilization:**
           - Maintain 85°F warm area
           - Offer electrolyte solution
           - Keep in quiet, dark location
        
        **Emergency Contacts:**
        • Avian Emergency Hotline: 1-800-AVIAN-VET
        • Animal Poison Control: (888) 426-4435
        """)
    else:
        st.markdown(f"""
        ## 📋 RECOMMENDED ACTION PLAN
        
        **For {severity.lower()} case:**
        
        1. **Monitoring Schedule:**
           - Check every 4-6 hours for changes
           - Record observations in log
           - Note any symptom progression
        
        2. **Home Care Instructions:**
           - Maintain optimal environment
           - Ensure proper nutrition
           - Reduce stress factors
        
        3. **Veterinary Follow-up:**
           - Schedule appointment within { '1-2 days' if severity == 'Moderate' else '3-5 days' }
           - Prepare symptom timeline
           - Collect droppings sample
        
        **When to escalate:**
        • Symptoms worsen or don't improve in 24-48 hours
        • New symptoms develop
        • Bird stops eating or drinking
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download Report
    report_content = f"""
    SQUAWK-A-THON AI DIAGNOSTIC REPORT
    ===================================
    Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
    Breed: {breed}
    Severity: {severity}
    
    SYMPTOM DESCRIPTION:
    {query}
    
    AI DIAGNOSIS:
    {diagnosis}
    
    MEDIA ANALYSIS:
    {media_analysis if media_analysis else 'No media analysis performed'}
    
    ACTION PLAN:
    {'URGENT: Veterinary attention required' if severity in ['Severe', 'Critical'] else 'Monitor and follow up as needed'}
    
    DISCLAIMER:
    AI-generated analysis for informational purposes only.
    Always consult with a certified avian veterinarian.
    """
    
    st.download_button(
        "📥 Download Complete Report",
        data=report_content,
        file_name=f"avian_diagnosis_{breed.replace(' ', '_')}_{time.strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- 9. EXAMPLE QUERIES ---
st.markdown("---")
with st.expander("📋 Example Cases for Testing"):
    st.markdown("""
    **Test these queries to see different AI responses:**
    
    1. **Feather Plucking:**
       *"My African Grey has been plucking feathers from his chest for 2 weeks. He only does it when I leave the room. Appetite and droppings normal."*
    
    2. **Respiratory Issues:**
       *"Cockatiel sneezing frequently, clear discharge from nostrils. Breathing sounds slightly raspy. Still eating but less active."*
    
    3. **Digestive Problems:**
       *"Sun Conure has watery green droppings for 3 days. Eating less than usual. Sleeping more. No vomiting."*
    
    4. **Behavioral Changes:**
       *"Macaw suddenly aggressive, biting when approached. Sleeping more. Feathers puffed up most of the day."*
    
    5. **Emergency Case:**
       *"Budgie found on cage floor, unable to perch. Breathing rapidly with tail bobbing. Not responding to touch."*
    """)

# --- 10. FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #aaa; font-size: 0.9em; padding: 20px;'>
<p><strong>🦜 Squawk-a-Thon AI Diagnostic System</strong></p>
<p>Powered by advanced AI models • Generates specific responses for each case</p>
<p><em>This tool provides AI-generated analysis only. Always consult a certified avian veterinarian for medical care.</em></p>
</div>
""", unsafe_allow_html=True)
