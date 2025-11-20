import streamlit as st
from chatbot import retrieve_and_answer

st.set_page_config(page_title="R&D Chunking Lab", page_icon="🧪", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🧪 Chunking Technique Evaluator")
st.markdown("Test different JSON chunking strategies live and check their efficiency scores.")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    technique = st.selectbox(
        "Select Chunking Strategy:",
        ("All (Compare)", "Flattened Key-Value", "Hierarchy-Aware", "Semantic Tree", "Graph-Based")
    )
    st.info(f"Currently Testing: **{technique}**")
    
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()

# Chat Input
query = st.chat_input("Ask a question about the car dataset...")

if query:
    # Add User Message
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    # Get Response & Metrics
    with st.spinner("Analyzing chunks and generating response..."):
        result = retrieve_and_answer(query, technique)
    
    # Add Assistant Message
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": result['answer'],
        "metrics": result['metrics']
    })

# Display History
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
        
        # Show Metrics only for Assistant
        if chat["role"] == "assistant" and chat.get("metrics"):
            m = chat["metrics"]
            
            # Color code the Score
            score_color = "green" if m['score'] > 1 else "orange" if m['score'] > -5 else "red"
            
            with st.expander(f"📊 Efficiency Report (Source: {m['source']})"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Relevance Score", f"{m['score']:.2f}", help="Higher is better. Cross-Encoder confidence.")
                col2.metric("Tokens Used", f"{m['total_cost_proxy']}", f"In: {m['input_tokens']} | Out: {m['output_tokens']}", help="Lower is cheaper.")
                col3.metric("Latency", f"{m['latency']}s", help="Time to retrieve and generate.")
                col4.caption(f"**Strategy Winner:**\n{m['source']}")