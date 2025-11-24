# 🧪 RAG Chunking Strategy R&D Lab (Nested JSON)

This project is a Research & Development framework designed to evaluate and optimize **Retrieval-Augmented Generation (RAG)** pipelines for **Nested JSON Data**.

It solves the common problem of *context loss* and *token inefficiency* when feeding structured data (like product catalogs or databases) into LLMs.

## 🚀 Key Features

- **Multi-Strategy Evaluation:** Dynamically switch between 4 different chunking techniques to test performance:
  1.  **Flattened Key-Value:** Best for specific attribute lookups (SQL-like behavior).
  2.  **Hierarchy-Aware:** Preserves JSON object structure.
  3.  **Semantic Tree:** Converts data to natural language sentences (Winner for Chatbots).
  4.  **Graph-Based:** Creates relationship triples for complex queries.
- **Live Metrics Dashboard:** Displays real-time efficiency stats in the chat UI:
  - **Relevance Score:** (MS-Marco Cross-Encoder) Confidence level of retrieval.
  - **Token Cost:** Input/Output token usage per query.
  - **Latency:** Retrieval time.
- **Automated Benchmarking:** Scripts to generate synthetic datasets and run ground-truth evaluations.
- **Smart Rate-Limiting:** Auto-retry logic to handle LLM API limits (Groq/OpenAI).

## 🛠️ Tech Stack

- **LLM:** Llama 3.1 (via Groq API)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Orchestration:** LangChain
- **UI:** Streamlit

## 📂 Project Structure

```bash
├── data/
│   ├── dataset_cars.json          # The raw nested JSON data
│   ├── dataset_ground_truth.json  # Q&A pairs for automated testing
│   ├── tech_flattened.txt         # Processed Chunks (Strategy 1)
│   ├── tech_semantic.txt          # Processed Chunks (Strategy 3 - Best)
│   └── ...
├── chatbot.py                     # Core RAG Logic + Metrics Calculation
├── experiment_chunking.py         # Script to convert JSON -> 4 Chunk Formats
├── evaluate_generation.py         # Script to run automated accuracy tests
├── generate_dataset.py            # Generates synthetic JSON data
├── main.py                        # Streamlit UI
└── requirements.txt               # Dependencies
```

## ⚡ Quick Start

### 1. Setup Environment
```bash
# Create Conda Environment
conda create -n rag_env python=3.10 -y
conda activate rag_env

# Install Dependencies
pip install -r requirements.txt
```

### 2. Configure Secrets
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_your_key_here
```

### 3. Generate Data & Chunks
Before running the app, generate the experimental data:
```bash
# 1. Create synthetic nested JSON
python generate_dataset.py

# 2. Process JSON into 4 different text formats
python experiment_chunking.py
```

### 4. Run the Lab
```bash
streamlit run main.py
```

## 📊 Research Findings

Based on our evaluation of 50 nested records:

| Technique | Best Used For | Accuracy | Token Efficiency |
| :--- | :--- | :--- | :--- |
| **Semantic Tree** | **General Chatbots / Summaries** | **High** | **Best (Lowest Cost)** |
| **Flattened** | Specific Filtering (Price/Year) | High | Medium |
| **Hierarchy** | Full Record Retrieval | High | Poor (High Cost) |
| **Graph** | Complex Relationship Mapping | Medium | Medium |

**Conclusion:**
For conversational AI agents interacting with JSON data, **Semantic Tree Chunking** is the recommended approach. It achieves a high Relevance Score (~4.8/10) while reducing token consumption by ~30% compared to raw JSON passing.

## 🛡️ automated Evaluation
To replicate the benchmark results:
```bash
python evaluate_generation.py
```
This will test all techniques against the Ground Truth dataset and output a CSV report.

## Link

[Steamlit](https://chunking-techniques-testing-ragchatbot.streamlit.app/)
