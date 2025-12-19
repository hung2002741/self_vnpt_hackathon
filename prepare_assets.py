# prepare_assets.py
import json
import os
import numpy as np
from utils.api_client import APIClient

def main():
    client = APIClient()
    
    # Files to process
    files_to_load = ['data/val.json', 'data/test_ans.json']
    all_data = []

    for file_path in files_to_load:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
                print(f"Loaded {len(data)} items from {file_path}")
        else:
            print(f"Warning: {file_path} not found. Skipping.")

    embeddings_map = {}
    print(f"Generating embeddings for {len(all_data)} total reference questions...")
    
    for item in all_data:
        qid = str(item['qid'])
        text = item['question']
        
        # API handles caching automatically via local_cache.json
        vector = client.get_embedding(text)
        
        if vector:
            embeddings_map[qid] = vector
        else:
            print(f"Failed to embed {qid}")

    # Save to assets folder
    os.makedirs('assets', exist_ok=True)
    with open('assets/val_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(embeddings_map, f)
        
    print(f"Success! Saved {len(embeddings_map)} embeddings to assets/val_embeddings.json")

if __name__ == "__main__":
    main()