import streamlit as st
# Import the updated function signature
from chatbot import retrieve_and_answer  

st.set_page_config(page_title="AWS Service Catalog Assistant", page_icon=":robot_face:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Chunking Strategy R&D")

# --- NEW: Sidebar for Technique Selection ---
st.sidebar.header("Configuration")
technique = st.sidebar.selectbox(
    "Select Chunking Technique:",
    (
        "All (Compare)", 
        "Flattened Key-Value", 
        "Hierarchy-Aware", 
        "Semantic Tree", 
        "Graph-Based"
    )
)

st.write(f"**Current Strategy:** {technique}")
st.write("Ask questions about the car dataset to see how this technique performs.")

# User input
query = st.text_input("Your question:")

if st.button("Submit Question"):
    if query.strip():
        # --- PASS THE SELECTED TECHNIQUE TO THE FUNCTION ---
        response = retrieve_and_answer(query, technique)
        st.session_state.chat_history.append({"user": query, "bot": response})
    else:
        st.warning("Please enter a question.")

if st.session_state.chat_history:
    st.subheader("Conversation History")
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Assistant:** {chat['bot']}")
        st.write("---")

if st.button("Clear Conversation"):
    st.session_state.chat_history = []
    st.rerun()