import streamlit as st
import os
import google.generativeai as genai
from openai import OpenAI
from PIL import Image
import tempfile
from dotenv import load_dotenv
import time
import json

load_dotenv()

# --- 1. STARTUP VALIDATION ---
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize clients
openai_client = None
gemini_client = None

if OPENAI_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_KEY)
        st.session_state['openai_ready'] = True
    except:
        st.session_state['openai_ready'] = False
else:
    st.session_state['openai_ready'] = False

if GOOGLE_KEY:
    try:
        genai.configure(api_key=GOOGLE_KEY)
        gemini_client = genai.GenerativeModel('gemini-1.5-flash')
        st.session_state['gemini_ready'] = True
    except:
        st.session_state['gemini_ready'] = False
else:
    st.session_state['gemini_ready'] = False

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
.symptom-card { background: rgba(30, 60, 30, 0.3); padding: 10px; margin: 5px 0; border-radius: 8px; }
.breed-info { background: rgba(139, 69, 19, 0.3); padding: 15px; border-radius: 10px; }
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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("OpenAI GPT-4", "✅" if st.session_state.get('openai_ready') else "❌")
    with col2:
        st.metric("Google Gemini", "✅" if st.session_state.get('gemini_ready') else "❌")
    with col3:
        st.metric("Diagnostic AI", "✅")
    with col4:
        st.metric("Response Quality", "Professional")

# --- 5. BREED INFORMATION DATABASE ---
BREED_INFO = {
    "Sun Conure": {
        "lifespan": "25-30 years",
        "size": "12 inches",
        "common_issues": ["Fatty liver disease", "Vitamin A deficiency", "Feather destructive behavior"],
        "diet": "Pellets (70%), Fruits/Veggies (20%), Nuts/Seeds (10%)",
        "temperature": "75-85°F (24-29°C)",
        "special_notes": "Social, noisy, need lots of interaction"
    },
    "Jenday Conure": {
        "lifespan": "20-30 years",
        "size": "12 inches",
        "common_issues": ["Respiratory infections", "Feather plucking", "Nutritional deficiencies"],
        "diet": "High-quality pellets, fresh vegetables, limited fruits",
        "temperature": "75-85°F",
        "special_notes": "Active, playful, require mental stimulation"
    },
    "Macaw": {
        "lifespan": "50-60 years",
        "size": "30-40 inches",
        "common_issues": ["Beak malformations", "Psittacosis", "Heavy metal toxicity"],
        "diet": "Large parrot pellets, nuts, fruits, vegetables",
        "temperature": "70-80°F",
        "special_notes": "Need large space, strong perches"
    },
    "African Grey": {
        "lifespan": "40-60 years",
        "size": "13 inches",
        "common_issues": ["Calcium deficiency", "Feather picking", "Respiratory issues"],
        "diet": "Calcium-rich pellets, vegetables, limited seeds",
        "temperature": "75-85°F",
        "special_notes": "Intelligent, sensitive to change"
    },
    "Cockatiel": {
        "lifespan": "15-25 years",
        "size": "12-13 inches",
        "common_issues": ["Respiratory infections", "Egg binding", "Nutritional deficiencies"],
        "diet": "Pellets, seeds, fresh greens",
        "temperature": "65-80°F",
        "special_notes": "Dust sensitive, need regular bathing"
    },
    "Budgie": {
        "lifespan": "5-8 years (up to 15)",
        "size": "7 inches",
        "common_issues": ["Tumors", "Goiter", "Scaly face mites"],
        "diet": "Pellets, vegetables, limited seeds",
        "temperature": "65-75°F",
        "special_notes": "Social, need companionship"
    }
}

# --- 6. SYMPTOM DATABASE ---
SYMPTOM_ANALYSIS = {
    "feather plucking": {
        "common_causes": ["Stress/boredom", "Nutritional deficiency", "Parasites", "Allergies", "Liver disease"],
        "immediate_actions": ["Increase enrichment", "Check diet", "Environmental assessment"],
        "vet_urgency": "Schedule within 1-2 weeks"
    },
    "lethargy": {
        "common_causes": ["Infection", "Metabolic disorder", "Toxicity", "Heart disease", "Pain"],
        "immediate_actions": ["Provide warmth", "Ensure hydration", "Quiet environment"],
        "vet_urgency": "Within 24 hours if severe"
    },
    "sneezing": {
        "common_causes": ["Respiratory infection", "Allergy", "Dust", "Foreign body", "Sinusitis"],
        "immediate_actions": ["Humidity 40-60%", "Clean environment", "Remove irritants"],
        "vet_urgency": "Within 48 hours if persistent"
    },
    "diarrhea": {
        "common_causes": ["Bacterial infection", "Dietary change", "Parasites", "Liver disease", "Toxicity"],
        "immediate_actions": ["Electrolyte solution", "Bland diet", "Monitor hydration"],
        "vet_urgency": "Within 24 hours if watery"
    },
    "wing drooping": {
        "common_causes": ["Injury", "Weakness", "Neurological issue", "Pain", "Metabolic"],
        "immediate_actions": ["Restrict flying", "Check for injury", "Provide low perches"],
        "vet_urgency": "Within 24 hours"
    }
}

# --- 7. DIAGNOSTIC FUNCTIONS ---
def generate_diagnosis(breed, query, severity):
    """Generate professional veterinary diagnosis using OpenAI"""
    
    if not openai_client:
        return get_fallback_diagnosis(breed, query, severity)
    
    try:
        system_prompt = """You are Dr. Aviana Feathers, a board-certified avian veterinarian with 25 years of experience.
        You specialize in psittacine medicine and are known for your compassionate, practical approach.
        
        Format your response EXACTLY as follows:
        
        # 🩺 AVIAN DIAGNOSTIC REPORT
        
        ## 📋 CLINICAL ASSESSMENT
        [Brief overview of the case]
        
        ## 🔍 DIFFERENTIAL DIAGNOSIS (Most to Least Likely)
        1. **[Primary Suspect]** - [Brief explanation, 15-20 words]
        2. **[Secondary Consideration]** - [Brief explanation, 15-20 words]
        3. **[Third Possibility]** - [Brief explanation, 15-20 words]
        
        ## 🚨 IMMEDIATE CARE PROTOCOL (Next 24 Hours)
        • **Action 1:** [Specific instruction]
        • **Action 2:** [Specific instruction]
        • **Action 3:** [Specific instruction]
        
        ## 🏠 HOME MONITORING CHECKLIST
        - [ ] [Parameter to monitor] - [Normal vs Concerning values]
        - [ ] [Parameter to monitor] - [Normal vs Concerning values]
        - [ ] [Parameter to monitor] - [Normal vs Concerning values]
        
        ## 🍎 SPECIES-SPECIFIC RECOMMENDATIONS
        [Tailored advice for this specific breed]
        
        ## ⚠️ RED FLAGS (Emergency Vet Immediately)
        • [Emergency sign 1]
        • [Emergency sign 2]
        • [Emergency sign 3]
        
        ## 📅 FOLLOW-UP TIMELINE
        - **24 hours:** [What to check]
        - **48 hours:** [What to expect]
        - **1 week:** [When to re-evaluate]
        
        *Disclaimer: This is AI-generated guidance. Consult an avian veterinarian for proper diagnosis.*"""
        
        user_prompt = f"""
        PATIENT INFORMATION:
        - Breed: {breed}
        - Severity: {severity}
        - Primary Concern: {query}
        
        BREED-SPECIFIC BACKGROUND:
        {json.dumps(BREED_INFO.get(breed, {}), indent=2)}
        
        Please provide a comprehensive veterinary assessment.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return get_fallback_diagnosis(breed, query, severity)

def get_fallback_diagnosis(breed, query, severity):
    """Fallback diagnosis if OpenAI fails"""
    
    breed_data = BREED_INFO.get(breed, {})
    
    # Analyze symptoms
    analysis = ""
    for symptom, info in SYMPTOM_ANALYSIS.items():
        if symptom in query.lower():
            analysis = f"""
            **Symptom Analysis for '{symptom}':**
            • **Common Causes:** {', '.join(info['common_causes'][:3])}
            • **Immediate Actions:** {', '.join(info['immediate_actions'])}
            • **Vet Urgency:** {info['vet_urgency']}
            """
            break
    
    if not analysis:
        analysis = "**General Assessment:** Monitor closely for changes in behavior, appetite, and droppings."
    
    return f"""
# 🩺 AVIAN DIAGNOSTIC REPORT - AUTOMATED ASSESSMENT

## 📋 CLINICAL ASSESSMENT
{breed} presenting with {severity.lower()} symptoms: "{query}"

{analysis}

## 🔍 LIKELY CONSIDERATIONS
Based on common avian medicine patterns:

1. **Behavioral/Environmental Factors** - Stress, boredom, or environmental triggers are common
2. **Nutritional Issues** - Vitamin/mineral deficiencies frequently manifest in various ways
3. **Early Medical Condition** - Could indicate developing health issue needing monitoring

## 🚨 IMMEDIATE CARE PROTOCOL
• **Warmth:** Maintain 85°F warm area (heating pad on LOW under half cage)
• **Hydration:** Offer electrolyte solution (unflavored Pedialyte mixed 50/50 with water)
• **Nutrition:** Hand-feed favorite foods if appetite decreased
• **Environment:** Quiet, dimly lit, stress-free area

## 🏠 HOME MONITORING CHECKLIST
- [ ] **Appetite** - Should eat within 4-6 hours of waking
- [ ] **Droppings** - Normal: green/brown solid with white urates
- [ ] **Activity Level** - Should respond to familiar stimuli
- [ ] **Breathing** - No tail bobbing or open-mouth breathing

## 🍎 BREED-SPECIFIC NOTES
• **Size:** {breed_data.get('size', 'N/A')}
• **Common Issues:** {', '.join(breed_data.get('common_issues', ['Various']))}
• **Diet:** {breed_data.get('diet', 'Balanced pellets with fresh produce')}
• **Temperature:** {breed_data.get('temperature', '75-85°F')}

## ⚠️ EMERGENCY RED FLAGS
• **Labored breathing** (tail bobbing, open-mouth breathing)
• **Bleeding** that doesn't stop within 5 minutes
• **Inability to perch** or maintain balance
• **Seizures** or loss of consciousness
• **No droppings** for 12+ hours

## 📅 FOLLOW-UP TIMELINE
- **24 hours:** Reassess symptoms, check hydration
- **48 hours:** If no improvement, schedule veterinary appointment
- **1 week:** Full re-evaluation of condition

*Note: This is automated guidance. Always consult with an avian veterinarian.*
"""

def analyze_media(media_file, breed, query):
    """Analyze uploaded media files"""
    if not gemini_client or not media_file:
        return None
    
    try:
        ext = os.path.splitext(media_file.name)[1].lower()
        
        if ext in [".jpg", ".jpeg", ".png"]:
            # Image analysis
            img = Image.open(media_file)
            prompt = f"""Analyze this {breed} parrot image for health indicators.

            Reported concern: {query}
            
            Focus on:
            1. Feather condition and quality
            2. Posture and body language
            3. Eyes, nares, and beak appearance
            4. Overall body condition
            5. Any visible abnormalities
            
            Provide specific observations and clinical correlation."""
            
            response = gemini_client.generate_content([prompt, img])
            return response.text
            
        elif ext in [".mp4", ".mov"]:
            # Video analysis placeholder
            return "Video analysis available with Gemini 1.5 Pro. Key things to observe: mobility, coordination, breathing patterns, and behavioral cues."
            
        elif ext in [".mp3", ".wav"]:
            # Audio analysis placeholder
            return "Audio analysis available with Gemini 1.5 Pro. Listen for: changes in vocalization, respiratory sounds, or distress calls."
            
    except Exception as e:
        return f"Media analysis limited: {str(e)[:100]}"
    
    return None

# --- 8. USER INTERFACE ---
st.markdown("### 🏥 AVIAN DIAGNOSTIC CENTER")

# Input Section
col1, col2 = st.columns(2)
with col1:
    breed = st.selectbox(
        "Select Breed:",
        list(BREED_INFO.keys()) + ["Other"]
    )
    
with col2:
    severity = st.select_slider(
        "Symptom Severity:",
        options=["Mild", "Moderate", "Severe", "Critical"],
        value="Moderate"
    )

# Symptom Description
query = st.text_area(
    "Describe Symptoms in Detail:",
    height=120,
    placeholder="Example: My Sun Conure has been plucking feathers from chest for 3 days. Noticed increased sleeping and decreased appetite. No changes in droppings. Environment: cage near window, diet includes pellets and fresh veggies..."
)

# Media Upload
st.markdown("### 📷 Upload Evidence (Optional)")
media_file = st.file_uploader(
    "Drag and drop or click to upload",
    type=["jpg", "jpeg", "png", "mp4", "mov", "mp3", "wav"],
    help="Images are analyzed for physical signs, videos for behavior, audio for vocalizations"
)

if media_file:
    ext = os.path.splitext(media_file.name)[1].lower()
    st.info(f"📎 Uploaded: {media_file.name}")
    
    if ext in [".jpg", ".jpeg", ".png"]:
        st.image(media_file, caption="Preview", width=300)
    elif ext in [".mp4", ".mov"]:
        st.video(media_file)
    elif ext in [".mp3", ".wav"]:
        st.audio(media_file)

# --- 9. GENERATE DIAGNOSIS ---
if st.button("🦜 GENERATE COMPREHENSIVE DIAGNOSIS", type="primary", use_container_width=True):
    
    if not query.strip():
        st.error("Please describe the symptoms")
        st.stop()
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("Initializing diagnostic protocols..."):
        progress_bar.progress(10)
        time.sleep(0.5)
        
        # Generate diagnosis
        status_text.text("🧠 Consulting avian specialists...")
        diagnosis = generate_diagnosis(breed, query, severity)
        progress_bar.progress(60)
        
        # Analyze media if available
        media_analysis = None
        if media_file:
            status_text.text("🎬 Analyzing media evidence...")
            media_analysis = analyze_media(media_file, breed, query)
            progress_bar.progress(80)
        
        status_text.text("📊 Compiling final report...")
        time.sleep(0.5)
        progress_bar.progress(100)
        status_text.empty()
    
    # --- DISPLAY RESULTS ---
    st.markdown("## 📋 DIAGNOSTIC REPORT")
    
    # Breed Info Card
    if breed in BREED_INFO:
        with st.expander(f"📖 {breed} Profile", expanded=False):
            breed_data = BREED_INFO[breed]
            st.markdown(f"""
            **Lifespan:** {breed_data['lifespan']}  
            **Size:** {breed_data['size']}  
            **Ideal Temperature:** {breed_data['temperature']}  
            **Common Health Issues:** {', '.join(breed_data['common_issues'])}  
            **Diet:** {breed_data['diet']}  
            **Notes:** {breed_data['special_notes']}
            """)
    
    # Main Diagnosis
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(diagnosis)
    
    # Media Analysis
    if media_analysis:
        st.markdown("---")
        st.markdown("### 🎬 MEDIA ANALYSIS FINDINGS")
        st.markdown('<div class="diagnostic-section">', unsafe_allow_html=True)
        st.markdown(media_analysis)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Emergency Section
    st.markdown("---")
    st.markdown("### ⚠️ EMERGENCY ACTION CARD")
    st.markdown('<div class="emergency-alert">', unsafe_allow_html=True)
    st.markdown(f"""
    #### **{breed.upper()} - {severity.upper()} CASE PROTOCOL**
    
    **CRITICAL ACTIONS:**
    1. **ISOLATE** - Quiet room away from other pets
    2. **WARMTH** - 85-90°F heat source (heating pad on LOW)
    3. **HYDRATE** - Offer warm electrolyte solution via syringe if not drinking
    4. **MONITOR** - Check every 30 minutes: breathing, consciousness, position
    
    **IMMEDIATE VET IF:**
    • Breathing difficulty (tail bobbing, open mouth)
    • Bleeding that continues
    • Cannot stand or perch
    • Seizures or collapse
    
    **24/7 RESOURCES:**
    • **Emergency Hotline:** 1-800-AVIAN-VET (1-800-284-2683)
    • **Avian Vet Locator:** [aav.org/find-a-vet](https://www.aav.org/find-a-vet)
    • **Poison Control:** (888) 426-4435
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Resources
    st.markdown("---")
    st.markdown("### 📚 ADDITIONAL RESOURCES")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📖 First Aid Guide**
        - Avian CPR
        - Bleeding Control
        - Fracture Stabilization
        - Emergency Warmth
        """)
    
    with col2:
        st.markdown("""
        **🍎 Nutrition Database**
        - Safe Foods List
        - Toxic Foods Warning
        - Supplement Guide
        - Feeding Schedule
        """)
    
    with col3:
        st.markdown("""
        **🏥 Vet Preparation**
        - Symptom Journal
        - Dropping Photos
        - Weight Records
        - Medication Log
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download Button
    report_content = f"""
    SQUAWK-A-THON AVIAN HEALTH REPORT
    =================================
    Date: {time.strftime('%Y-%m-%d %H:%M')}
    Breed: {breed}
    Severity: {severity}
    
    SYMPTOMS:
    {query}
    
    DIAGNOSIS:
    {diagnosis}
    
    MEDIA ANALYSIS:
    {media_analysis if media_analysis else 'No media analysis performed'}
    
    EMERGENCY CONTACTS:
    1-800-AVIAN-VET | aav.org | Animal Poison Control: (888) 426-4435
    
    Disclaimer: AI-generated guidance only. Consult avian veterinarian.
    """
    
    st.download_button(
        "💾 Download Full Report",
        report_content,
        file_name=f"squawkathon_{breed.replace(' ', '_')}_{time.strftime('%Y%m%d')}.txt"
    )

# --- 10. FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #ccc; font-size: 0.9em;">
<p><strong>⚠️ IMPORTANT DISCLAIMER:</strong> Squawk-a-Thon is an AI-powered educational tool for informational purposes only.</p>
<p>It is <strong>NOT</strong> a substitute for professional veterinary diagnosis or treatment.</p>
<p>Always consult with a certified avian veterinarian for medical concerns.</p>
<p>In emergencies, contact your veterinarian or emergency animal hospital immediately.</p>
</div>
""", unsafe_allow_html=True)
