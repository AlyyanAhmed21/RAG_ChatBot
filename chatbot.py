import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# CHANGED: Use RecursiveCharacterTextSplitter (It handles large chunks better)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- CONFIGURATION ---
TECHNIQUE_MAP = {
    "Flattened Key-Value": "tech_flattened.txt",
    "Hierarchy-Aware": "tech_hierarchy.txt",
    "Semantic Tree": "tech_semantic.txt",
    "Graph-Based": "tech_graph.txt"
}

# Initialize Embeddings & Reranker
load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, documents, top_k=3):
    if not documents: return []
    pairs = [(query, doc.page_content) for doc in documents]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze(-1)
    
    # Sort by score
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    # --- CHANGE: Save the score to the document metadata ---
    top_results = []
    for doc, score in ranked[:top_k]:
        doc.metadata['relevance_score'] = float(score) # Save score here
        top_results.append(doc)
        
    return top_results

def load_documents(data_path="data", selected_technique="All (Compare)"):
    docs = []
    if not os.path.exists(data_path): return []

    target_file = TECHNIQUE_MAP.get(selected_technique, None)

    for file in os.listdir(data_path):
        if target_file and file != target_file:
            continue
        if not target_file and not file.endswith(".txt"):
            continue

        file_path = os.path.join(data_path, file)
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            loaded = loader.load()
            for doc in loaded:
                doc.metadata['source'] = file
            docs.extend(loaded)
            print(f"✅ Loaded: {file}")
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")

    return docs

def create_vector_store(selected_technique):
    print("⏳ Loading and splitting documents...")
    raw_docs = load_documents(selected_technique=selected_technique)
    if not raw_docs: 
        return None
    
    # --- CHANGED: Increased Chunk Size & Better Splitter ---
    # This prevents the warnings and allows the full Graph/Semantic descriptions to stay intact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,   # Increased from 1000 to 2000 to fit your large chunks
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""] # Try to split by paragraphs first
    )
    split_docs = text_splitter.split_documents(raw_docs)
    print(f"🔹 Created {len(split_docs)} vector chunks.")
    
    return FAISS.from_documents(split_docs, embeddings)

def get_llm():
    # --- CHANGED: Switched to 8B Model ---
    return ChatGroq(model_name="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(query, context, tech_name):
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are analyzing data processed using the '{tech_name}' chunking technique.\n"
        "Use ONLY the provided context to answer. Be concise.\n"
        "<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>"
    )

def retrieve_and_answer(query, selected_technique):
    print(f"\n--- 🟢 Processing with Strategy: {selected_technique} ---")
    
    # 1. Create vector store
    vector_store = create_vector_store(selected_technique)
    
    if not vector_store:
        return f"No data found for technique: {selected_technique}. Run experiment_chunking.py first."

    # 2. Retrieve (Get top 10 candidates)
    results = vector_store.similarity_search_with_score(query, k=10)
    retrieved_docs = [doc for doc, _ in results]

    # 3. Rerank (Get top 3 and assign scores)
    top_docs = rerank(query, retrieved_docs, top_k=3)
    
    combined_context = "\n".join([d.page_content for d in top_docs])
    
    # --- CHANGED: R&D DEBUG PRINT ---
    print("\n📊 R&D RANKING REPORT:")
    print(f"❓ Query: {query}")
    print("-" * 60)
    for i, d in enumerate(top_docs):
        score = d.metadata.get('relevance_score', 0.0)
        source = d.metadata.get('source', 'Unknown')
        # Print formatted row
        print(f"🏆 Rank {i+1} | Score: {score:.4f} | Tech: {source}")
    print("-" * 60 + "\n")
    # --------------------------------

    # 4. Generate
    llm = get_llm()
    prompt = build_prompt(query, combined_context, selected_technique)
    response = llm.invoke(prompt).content
    
    return response