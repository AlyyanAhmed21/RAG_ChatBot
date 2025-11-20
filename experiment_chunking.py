import json
import os
from langchain_text_splitters import RecursiveJsonSplitter

# --- 1. FLATTENED KEY-VALUE CHUNKING ---
def flatten_json(y):
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            for i, a in enumerate(x):
                flatten(a, name + str(i) + '_')
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

def technique_flattened(data):
    """Converts nested JSON to single-depth Key-Value pairs."""
    text_output = []
    for car in data:
        flat = flatten_json(car)
        # Convert to string representation
        lines = [f"{k}: {v}" for k, v in flat.items()]
        block = "--- CAR ENTRY (FLATTENED) ---\n" + "\n".join(lines)
        text_output.append(block)
    return "\n\n".join(text_output)

# --- 2. HIERARCHY-AWARE CHUNKING ---
def technique_hierarchy(data):
    """Uses LangChain's RecursiveJsonSplitter to preserve structure."""
    # We process one car at a time to ensure we don't lose context
    splitter = RecursiveJsonSplitter(max_chunk_size=300)
    text_output = []
    
    for car in data:
        # create_documents returns Document objects, we want text
        docs = splitter.create_documents(texts=[car])
        for doc in docs:
            text_output.append(f"--- CAR ENTRY (HIERARCHY) ---\n{doc.page_content}")
            
    return "\n\n".join(text_output)

# --- 3. SEMANTIC TREE CHUNKING ---
def technique_semantic(data):
    """
    Re-writes the JSON into natural language paragraphs based on 
    logical grouping (Specs, Location, Pricing).
    """
    text_output = []
    for car in data:
        # Handle missing keys gracefully
        make = car.get('make', 'Unknown Make')
        model = car.get('model', 'Unknown Model')
        year = car.get('year', 'N/A')
        price = car.get('price', 'N/A')
        city = car.get('location', {}).get('city', 'Unknown City')
        
        # Grouping Specs
        engine = car.get('engine', {})
        features = ", ".join(car.get('features', []))
        
        # Construct Semantic Sentence Tree
        content = (
            f"--- CAR ENTRY (SEMANTIC) ---\n"
            f"Vehicle Identification: This is a {year} {make} {model} located in {city}.\n"
            f"Technical Specifications: It features a {engine.get('type', 'standard')} engine "
            f"with {engine.get('transmission', 'manual')} transmission.\n"
            f"Features & Add-ons: The car comes equipped with: {features}.\n"
            f"Market Value: The listed price for this vehicle is {price}."
        )
        text_output.append(content)
    return "\n\n".join(text_output)

# --- 4. GRAPH-BASED CHUNKING (Triples) ---
def technique_graph(data):
    """
    Converts data into Subject -> Predicate -> Object triples.
    Good for questions like 'Which cars have Sunroof?'
    """
    text_output = []
    for car in data:
        car_id = f"{car.get('make')} {car.get('model')} ({car.get('id')})"
        triples = []
        
        # Helper to add triple
        def add_t(pred, obj):
            triples.append(f"({car_id}) --[{pred}]--> ({obj})")

        add_t("has_year", car.get('year'))
        add_t("located_in", car.get('location', {}).get('city'))
        add_t("costs", car.get('price'))
        
        for feature in car.get('features', []):
            add_t("has_feature", feature)
            
        block = "--- CAR ENTRY (GRAPH TRIPLES) ---\n" + "\n".join(triples)
        text_output.append(block)
        
    return "\n\n".join(text_output)

def run_experiment():
    data_path = "data"
    # CHANGED: Point to the new generated dataset
    json_file = os.path.join(data_path, "dataset_cars.json")
    
    if not os.path.exists(json_file):
        print("❌ dataset_cars.json not found. Run generate_dataset.py first.")
        return

    print(f"📂 Loading {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing {len(data)} car entries...")

    techniques = [
        ("tech_flattened.txt", technique_flattened),
        ("tech_hierarchy.txt", technique_hierarchy),
        ("tech_semantic.txt", technique_semantic),
        ("tech_graph.txt", technique_graph)
    ]

    for filename, func in techniques:
        try:
            content = func(data)
            out_path = os.path.join(data_path, filename)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Generated: {filename}")
        except Exception as e:
            print(f"❌ Error {filename}: {e}")

if __name__ == "__main__":
    run_experiment()