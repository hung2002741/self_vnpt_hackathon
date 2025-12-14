# evaluate_dev.py
import json
import pandas as pd
import numpy as np
import argparse
import time
from collections import Counter

# Import necessary components from your pipeline
# We import specific variables/functions to reuse logic
import predict
from predict import (
    client, 
    construct_prompt, 
    extract_answer, 
    format_choices,
    VAL_VECTORS, 
    VAL_LOOKUP,
    cosine_similarity
)

def get_safe_examples(question_text, current_qid, top_k=2):
    """
    Finds similar examples but EXCLUDES the current question ID.
    This prevents Data Leakage (cheating) when evaluating on the validation set.
    """
    if not VAL_VECTORS: return []
    
    # 1. Embed current question
    # Note: In a real heavy eval, you might want to cache these or read from the file directly 
    # if you already have the embedding for the current_qid.
    curr_vec = client.get_embedding(question_text)
    if not curr_vec: return []

    # 2. Compare with all val vectors
    scores = []
    for qid, vec in VAL_VECTORS.items():
        # CRITICAL: Skip the question we are currently trying to solve!
        if str(qid) == str(current_qid):
            continue
            
        score = cosine_similarity(curr_vec, vec)
        scores.append((score, qid))
    
    # 3. Sort and pick top K
    scores.sort(key=lambda x: x[0], reverse=True)
    best_examples = []
    
    for score, qid in scores[:top_k]:
        ex = VAL_LOOKUP.get(qid)
        if ex:
            best_examples.append(ex)
            
    return best_examples

def evaluate_single(row, use_rag=True):
    """
    Solves a single row using the logic from predict.py, 
    but uses get_safe_examples to avoid leakage.
    """
    qid = row['qid']
    ground_truth = row['answer']
    
    # 1. RAG (Safe Mode)
    examples = []
    if use_rag:
        examples = get_safe_examples(row['question'], current_qid=qid)
    
    # 2. Prompt Construction
    full_prompt = construct_prompt(row['question'], row['choices'], examples)
    messages = [{"role": "user", "content": full_prompt}]
    
    # 3. Model Selection
    model = "large" if len(row['question']) > 800 else "small"
    
    print(f"[{qid}] Asking {model.upper()}...", end=" ")
    
    # 4. API Call (Reusing parameters from predict.py)
    response = client.call_chat(
        messages, 
        model_type=model, 
        n=3, # Majority voting
        temperature=0.7, 
        top_p=0.9
    )
    
    votes = []
    if response and 'choices' in response:
        for choice in response['choices']:
            content = choice['message']['content']
            ans = extract_answer(content)
            if ans:
                votes.append(ans)
    
    final_answer = "C" # Default fallback
    if votes:
        count = Counter(votes)
        final_answer, freq = count.most_common(1)[0]
        print(f"-> Votes: {votes} -> Pred: {final_answer} | True: {ground_truth}", end=" ")
    else:
        print(f"-> Failed (No output) -> Pred: C | True: {ground_truth}", end=" ")

    is_correct = (final_answer == ground_truth)
    if is_correct:
        print("✅")
    else:
        print("❌")
        
    return {
        "qid": qid,
        "question": row['question'],
        "predicted": final_answer,
        "ground_truth": ground_truth,
        "is_correct": is_correct,
        "choices": str(row['choices'])
    }

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0, help='Number of questions to evaluate')
    parser.add_argument('--no-rag', action='store_true', help='Disable RAG')
    args = parser.parse_args()

    # Load Validation Data (Ground Truth)
    print("📂 Loading validation data...")
    with open('data/val.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    # Filter if limit is set
    if args.limit > 0:
        val_data = val_data[:args.limit]
        print(f"🧪 Evaluation limited to first {args.limit} examples.")

    results = []
    correct_count = 0
    total = len(val_data)

    print(f"🚀 Starting evaluation on {total} questions...")

    for i, item in enumerate(val_data):
        res = evaluate_single(item, use_rag=not args.no_rag)
        results.append(res)
        if res['is_correct']:
            correct_count += 1
            
        # Optional: Save progress every 10 questions
        if (i + 1) % 10 == 0:
            current_acc = (correct_count / (i + 1)) * 100
            print(f"--- Progress: {i+1}/{total} | Current Acc: {current_acc:.2f}% ---")

    accuracy = (correct_count / total) * 100
    print(f"==============================")
    print(f"Final Accuracy: {accuracy:.2f}% ({correct_count}/{total})")
    
    # Save detailed analysis
    df = pd.DataFrame(results)
    df.to_csv('evaluation_report.csv', index=False)
    
    # Save only failures for debugging
    failures = df[df['is_correct'] == False]
    failures.to_csv('evaluation_failures.csv', index=False)
    print("📂 Saved 'evaluation_report.csv' and 'evaluation_failures.csv'")

if __name__ == "__main__":
    evaluate()