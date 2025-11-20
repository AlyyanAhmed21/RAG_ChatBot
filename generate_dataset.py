import json
import random
import os

DATA_DIR = "data"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

MAKES = {
    "Toyota": ["Corolla", "Yaris", "Fortuner", "Camry"],
    "Honda": ["Civic", "City", "BR-V", "Accord"],
    "Suzuki": ["Alto", "Cultus", "Swift", "WagonR"]
}
CITIES = ["Lahore", "Karachi", "Islamabad", "Multan"]
FEATURES = ["Sunroof", "Leather Seats", "Cruise Control", "Navigation", "Alloy Rims", "Heated Seats"]

def generate_cars(n=50):
    cars = []
    ground_truth_qa = []

    for i in range(1, n + 1):
        make = random.choice(list(MAKES.keys()))
        model = random.choice(MAKES[make])
        city = random.choice(CITIES)
        price = random.randint(15, 80) * 100000 # 15 lacs to 80 lacs
        
        car_id = f"CAR-{1000+i}" # Unique ID for Ground Truth tracking
        
        car = {
            "id": car_id,
            "make": make,
            "model": model,
            "year": random.randint(2018, 2024),
            "price": f"PKR {price}",
            "location": {
                "city": city,
                "province": "Punjab" if city in ["Lahore", "Multan"] else "Sindh" if city == "Karachi" else "Federal"
            },
            "specs": {
                "engine": f"{random.choice(['1.0L', '1.3L', '1.5L', '1.8L'])}",
                "transmission": random.choice(["Automatic", "Manual", "CVT"])
            },
            "features": random.sample(FEATURES, k=3)
        }
        cars.append(car)

        # --- Create a Ground Truth Question for this specific car ---
        # We ask a question that requires finding THIS specific ID.
        question = {
            "question": f"What is the price and transmission of the {make} {model} with ID {car_id}?",
            "expected_id": car_id, # We will check if the retrieval finds this ID
            "complexity": "Look-up"
        }
        ground_truth_qa.append(question)

    return cars, ground_truth_qa

if __name__ == "__main__":
    cars_data, qa_data = generate_cars(50)
    
    # Save Cars
    with open(os.path.join(DATA_DIR, "dataset_cars.json"), "w") as f:
        json.dump(cars_data, f, indent=2)
        
    # Save Ground Truth
    with open(os.path.join(DATA_DIR, "dataset_ground_truth.json"), "w") as f:
        json.dump(qa_data, f, indent=2)
        
    print(f"✅ Generated 50 Cars in 'dataset_cars.json'")
    print(f"✅ Generated 50 Ground Truth Questions in 'dataset_ground_truth.json'")