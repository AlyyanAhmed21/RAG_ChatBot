import os
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    
    ranked = sorted(zip(documents, scores.tolist()), key=lambda x: x[1], reverse=True)
    
    top_results = []
    for doc, score in ranked[:top_k]:
        doc.metadata['relevance_score'] = score
        top_results.append(doc)
    return top_results

def load_documents(data_path="data", selected_technique="All (Compare)"):
    docs = []
    if not os.path.exists(data_path): return []
    target_file = TECHNIQUE_MAP.get(selected_technique, None)

    for file in os.listdir(data_path):
        if target_file and file != target_file: continue
        if not target_file and not file.endswith(".txt"): continue

        try:
            loader = TextLoader(os.path.join(data_path, file), encoding='utf-8')
            loaded = loader.load()
            for doc in loaded: doc.metadata['source'] = file
            docs.extend(loaded)
        except: pass
    return docs

def create_vector_store(selected_technique):
    raw_docs = load_documents(selected_technique=selected_technique)
    if not raw_docs: return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(raw_docs)
    return FAISS.from_documents(split_docs, embeddings)

def get_llm():
    return ChatGroq(model_name="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

def build_prompt(query, context, tech_name):
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"Using context from '{tech_name}', answer the question.\n"
        "<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>"
    )

# --- MODIFIED FUNCTION TO RETURN METRICS ---
def retrieve_and_answer(query, selected_technique):
    start_time = time.time()
    metrics = {}
    
    # 1. Vector Store Creation
    vector_store = create_vector_store(selected_technique)
    if not vector_store: 
        return {"answer": "No data found.", "metrics": None}

    # 2. Retrieve
    results = vector_store.similarity_search_with_score(query, k=10)
    retrieved_docs = [doc for doc, _ in results]

    # 3. Rerank
    top_docs = rerank(query, retrieved_docs, top_k=3)
    
    # Get Top Score and Source
    top_score = top_docs[0].metadata.get('relevance_score', 0.0) if top_docs else 0.0
    top_source = top_docs[0].metadata.get('source', 'Unknown') if top_docs else "None"
    
    combined_context = "\n".join([d.page_content for d in top_docs])
    
    # Calculate Input Tokens (Approx: 4 chars = 1 token)
    input_tokens = len(combined_context) // 4
    
    # 4. Generate
    llm = get_llm()
    prompt = build_prompt(query, combined_context, selected_technique)
    
    try:
        response = llm.invoke(prompt)
        answer_text = response.content
        
        # Try to get exact token usage from Groq API, else fallback to calculation
        usage = response.response_metadata.get('token_usage', {})
        final_input_tokens = usage.get('prompt_tokens', input_tokens)
        output_tokens = usage.get('completion_tokens', len(answer_text)//4)
        
    except Exception as e:
        return {"answer": f"LLM Error: {str(e)}", "metrics": None}

    end_time = time.time()
    latency = round(end_time - start_time, 2)

    metrics = {
        "score": top_score,
        "source": top_source,
        "latency": latency,
        "input_tokens": final_input_tokens,
        "output_tokens": output_tokens,
        "total_cost_proxy": final_input_tokens + output_tokens
    }
    
    return {"answer": answer_text, "metrics": metrics}