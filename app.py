import streamlit as st
from brain import AvianBrain
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Squawk-A-Thon AI", page_icon="🦜", layout="centered")

# --- CUSTOM BEAUTIFICATION ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #f0fdf4 0%, #fffbeb 100%);
    }
    
    /* Title Styling */
    .title-text {
        font-family: 'Helvetica Neue', sans-serif;
        color: #166534;
        font-weight: 800;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0;
    }
    
    /* Subtitle */
    .subtitle-text {
        color: #92400e;
        text-align: center;
        font-style: italic;
        margin-bottom: 2rem;
    }

    /* Input Card */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #bbf7d0 !important;
    }

    /* Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #166534 0%, #15803d 100%);
        color: white;
        border-radius: 12px;
        padding: 0.5rem 2rem;
        font-size: 1.2rem;
        font-weight: bold;
        border: none;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    }

    /* Result Box */
    .result-card {
        background-color: white;
        padding: 2rem;
        border-radius: 20px;
        border-left: 10px solid #fbbf24;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- UI LAYOUT ---
st.markdown('<h1 class="title-text">🦜 Squawk-A-Thon</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">The World\'s First Multi-Agent Avian Health Assistant</p>', unsafe_allow_html=True)

# Logo Placement
col1, col2, col3 = st.columns([1,2,1])
with col2:
    # Ensure you have your logo in the assets folder
    try:
        st.image("assets/logo.png", use_container_width=True)
    except:
        st.info("💡 Place your logo.png in the assets folder!")

# User Input Section
with st.container():
    st.write("### 🔍 Tell us about your bird")
    bird_species = st.selectbox("Bird Species", ["Sun Conure", "Jenday Conure", "Budgie", "Cockatiel", "African Grey", "Other"])
    user_input = st.text_area("Describe the symptoms or behavior (e.g., 'not eating, sleeping more than usual')", height=100)
    
    if st.button("RUN HEALTH DIAGNOSTIC 🚀"):
        if user_input:
            with st.status("🦜 Agents are conferring...", expanded=True) as status:
                st.write("Avian Pathologist is analyzing symptoms...")
                brain = AvianBrain()
                
                # Running the Crew Logic
                full_input = f"Species: {bird_species}. Symptoms: {user_input}"
                response = brain.process_request(full_input)
                
                status.update(label="Diagnostic Complete!", state="complete", expanded=False)

            # Displaying the Result in a Beautiful Card
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("## 📋 Specialist Report")
            st.markdown(response)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.download_button("Download Care Plan", str(response), file_name="care_plan.txt")
        else:
            st.warning("Please enter some symptoms first!")

# Footer
st.markdown("---")
st.caption("⚠️ Disclaimer: This AI is a decision-support tool and does not replace a physical exam by a certified Avian Veterinarian.")
