# prepare_assets.py
import json
import os
import numpy as np
from utils.api_client import APIClient

def main():
    client = APIClient()
    
    # 1. Load Validation Data
    with open('data/val.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
        
    embeddings_map = {}
    
    print(f"Generating embeddings for {len(val_data)} validation questions...")
    
    for item in val_data:
        qid = item['qid']
        text = item['question']
        
        # Call API (it handles caching automatically)
        vector = client.get_embedding(text)
        
        if vector:
            embeddings_map[qid] = vector
        else:
            print(f"Failed to embed {qid}")

    # 2. Save to assets folder
    os.makedirs('assets', exist_ok=True)
    with open('assets/val_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(embeddings_map, f)
        
    print("Success! Saved to assets/val_embeddings.json")

if __name__ == "__main__":
    main()