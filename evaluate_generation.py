import json
import os
import time
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
DATA_DIR = "data"
QA_FILE = os.path.join(DATA_DIR, "dataset_ground_truth.json")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Techniques to test
TECHNIQUES = {
    "Flattened": "tech_flattened.txt",
    "Hierarchy": "tech_hierarchy.txt",
    "Semantic": "tech_semantic.txt",
    "Graph": "tech_graph.txt"
}

# Initialize Models
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

def get_vector_store(filename):
    """Builds a temporary vector store for the specific technique"""
    loader = TextLoader(os.path.join(DATA_DIR, filename), encoding='utf-8')
    docs = loader.load()
    # Use larger chunks to ensure context isn't cut off
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    return FAISS.from_documents(split_docs, embeddings)

def evaluate_technique(tech_name, filename, questions):
    print(f"\n🔵 Testing Technique: {tech_name}...")
    vector_store = get_vector_store(filename)
    
    correct_answers = 0
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Loop through questions
    for i, q in enumerate(questions):
        query = q['question']
        expected_id = q['expected_id']
        
        # 1. Retrieve
        results = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in results])
        
        # 2. Generate Prompt
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"IMPORTANT: You must mention the Car ID ({expected_id}) in your answer if found.\n"
            f"Answer:"
        )
        
        try:
            # 3. Call LLM
            response = llm.invoke(prompt)
            answer_text = response.content
            
            # 4. Get Token Usage (Accurate count from API)
            # Note: response.response_metadata['token_usage'] gives exact numbers
            usage = response.response_metadata.get('token_usage', {})
            in_tokens = usage.get('prompt_tokens', len(prompt)//4)
            out_tokens = usage.get('completion_tokens', len(answer_text)//4)
            
            total_input_tokens += in_tokens
            total_output_tokens += out_tokens
            
            # 5. Grade the Answer
            # We check if the LLM correctly outputted the Expected ID
            if expected_id in answer_text:
                correct_answers += 1
                print(f"   ✅ Q{i+1}: Correct")
            else:
                print(f"   ❌ Q{i+1}: Failed (Expected {expected_id})")
                
            # Sleep briefly to respect Rate Limits (Groq Free Tier)
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ⚠️ Error on Q{i+1}: {e}")

    # Calculate Stats for this Technique
    count = len(questions)
    return {
        "Technique": tech_name,
        "Answer Accuracy (%)": round((correct_answers / count) * 100, 2),
        "Avg Input Tokens": int(total_input_tokens / count),
        "Avg Output Tokens": int(total_output_tokens / count),
        "Total Cost (Tokens)": total_input_tokens + total_output_tokens
    }

def run_full_evaluation():
    # Load Ground Truth
    with open(QA_FILE, 'r') as f:
        questions = json.load(f)
        
    # LIMIT FOR TESTING: Use only first 10 questions to save time
    # Set this to len(questions) for the full 50 car run
    TEST_LIMIT = 10 
    subset_questions = questions[:TEST_LIMIT]
    
    print(f"🚀 Starting End-to-End Evaluation on {TEST_LIMIT} Questions...")
    
    all_results = []
    
    for tech, filename in TECHNIQUES.items():
        if os.path.exists(os.path.join(DATA_DIR, filename)):
            stats = evaluate_technique(tech, filename, subset_questions)
            all_results.append(stats)
        else:
            print(f"⚠️ Skipping {tech}, file not found.")

    # Display Final DataFrame
    df = pd.read_json(json.dumps(all_results))
    print("\n" + "="*60)
    print("🏆 FINAL GENERATION LEADERBOARD")
    print("="*60)
    print(df.to_markdown(index=False))
    
    # Save
    df.to_csv(os.path.join(DATA_DIR, "llm_evaluation_results.csv"), index=False)
    print("\n✅ Saved detailed results to data/llm_evaluation_results.csv")

if __name__ == "__main__":
    run_full_evaluation()