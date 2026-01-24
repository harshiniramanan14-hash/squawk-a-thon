import streamlit as st
from brain import get_rag_context, crew_ai_response

st.set_page_config(
    page_title="Squawk-A-Thon 🦜",
    page_icon="🦜",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #0f2027, #203a43, #2c5364);
}
h1, h2, h3, p, label {
    color: #eaffea !important;
}
</style>
""", unsafe_allow_html=True)

st.image(
    "assets/ChatGPT Image Jan 23, 2026, 07_37_38 PM.png",
    width=120
)

st.title("🦜 Squawk-A-Thon")
st.caption("AI Assistant for Avian Healthcare & Behavioral Analysis")

breed = st.selectbox(
    "Select Bird Breed",
    [
        "Sun Conure",
        "Jenday Conure",
        "African Grey",
        "Cockatiel",
        "Budgerigar",
        "Macaw",
        "Lovebird",
        "cockatoo",
        "Other"
    ]
)

query = st.text_area(
    "Describe the health or behavior concern",
    placeholder="My bird screams excessively and is plucking feathers..."
)

st.subheader("📂 Upload Media (Optional)")
st.file_uploader("Upload Audio", type=["wav", "mp3"])
st.file_uploader("Upload Video", type=["mp4", "mov"])
st.file_uploader("Upload Reports", type=["pdf"], accept_multiple_files=True)

if st.button("Analyze 🧠"):
    if not query.strip():
        st.warning("Please describe the issue.")
    else:
        with st.spinner("Consulting the flock..."):
            try:
                context = get_rag_context(query)
            except ValueError as e:
                st.warning(str(e))
                context = "No research documents available. Answer based on general avian knowledge."

            response = crew_ai_response(
                query + "\n\nRelevant Research Context:\n" + context,
                breed
            )

        st.success("Analysis Complete")
        st.markdown(response)

