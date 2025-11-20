import json
import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup
DATA_DIR = "data"
QA_FILE = os.path.join(DATA_DIR, "dataset_ground_truth.json")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

TECHNIQUES = {
    "Flattened": "tech_flattened.txt",
    "Hierarchy": "tech_hierarchy.txt",
    "Semantic": "tech_semantic.txt",
    "Graph": "tech_graph.txt"
}

def get_token_count(text):
    # Rough estimation: 4 chars ~= 1 token
    return len(text) / 4

def evaluate():
    print("🚀 Starting Ground Truth Evaluation...\n")
    
    # Load Questions
    with open(QA_FILE, 'r') as f:
        questions = json.load(f)
    
    results = []

    for tech_name, filename in TECHNIQUES.items():
        print(f"🔵 Testing Technique: {tech_name}...")
        
        # 1. Create Vector Store for this technique
        loader = TextLoader(os.path.join(DATA_DIR, filename), encoding='utf-8')
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(split_docs, embeddings)
        
        hits = 0
        total_tokens = 0
        
        # 2. Run all 50 Questions
        for q in questions:
            query = q['question']
            expected_id = q['expected_id']
            
            # Retrieve Top 1 Chunk
            retrieved = vector_store.similarity_search(query, k=1)
            top_chunk = retrieved[0].page_content
            
            # CHECK 1: ACCURACY (Does the chunk contain the ID?)
            # Since we generated the ID in the text (e.g., "ID CAR-1001"), this checks recall.
            if expected_id in top_chunk:
                hits += 1
            
            # CHECK 2: EFFICIENCY (Token Count)
            total_tokens += get_token_count(top_chunk)
            
        # Calculate Metrics
        accuracy = (hits / len(questions)) * 100
        avg_tokens = total_tokens / len(questions)
        
        results.append({
            "Technique": tech_name,
            "Accuracy (%)": accuracy,
            "Avg Tokens per Chunk": int(avg_tokens),
            "Efficiency Score": round(accuracy / avg_tokens, 2) # Higher is better
        })
        
    # Display Final Report
    df = pd.read_json(json.dumps(results))
    print("\n" + "="*50)
    print("🏆 FINAL R&D RESULTS")
    print("="*50)
    print(df.to_markdown(index=False))
    
    # Save to CSV
    df.to_csv("final_evaluation_results.csv", index=False)
    print("\n✅ Results saved to 'final_evaluation_results.csv'")

if __name__ == "__main__":
    evaluate()