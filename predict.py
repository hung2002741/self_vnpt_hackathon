# predict.py
import json
import os
import pandas as pd
import numpy as np
from utils.api_client import APIClient
import argparse  
from collections import Counter
import re
import logging

# --- SETUP LOGGING ---
logging.basicConfig(
    filename='execution_debug.log', 
    filemode='w', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

client = APIClient()

# Load Assets
VAL_DATA = []
VAL_VECTORS = {}
try:
    with open('data/val.json', 'r', encoding='utf-8') as f:
        VAL_DATA = json.load(f)
        VAL_LOOKUP = {str(item['qid']): item for item in VAL_DATA}
        
    with open('assets/val_embeddings.json', 'r', encoding='utf-8') as f:
        VAL_VECTORS = json.load(f)
except Exception as e:
    logging.warning(f"Asset Load Error: {e}")

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_similar_examples(question_text, top_k=2):
    if not VAL_VECTORS: return []
    curr_vec = client.get_embedding(question_text)
    if not curr_vec: return []

    scores = []
    for qid, vec in VAL_VECTORS.items():
        score = cosine_similarity(curr_vec, vec)
        scores.append((score, qid))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    best_examples = []
    
    for score, qid in scores[:top_k]:
        ex = VAL_LOOKUP.get(str(qid)) or VAL_LOOKUP.get(int(qid))
        if ex:
            best_examples.append(ex)
    return best_examples

def format_choices(choices_list):
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    formatted = []
    for i, choice in enumerate(choices_list):
        if i < len(labels):
            formatted.append(f"{labels[i]}. {choice}")
    return "\n".join(formatted)

def extract_answer(text):
    """
    Robust extraction logic. 
    1. Search from the END of the text upwards.
    2. Prioritize explicit formats like "Đáp án: A".
    3. Fallback to looking for standalone letters.
    """
    if not text: return None
    
    # Normalize
    text = text.replace('*', '') # Remove bolding
    lines = text.split('\n')
    lines.reverse() # Process from bottom up
    
    # Pattern 1: Strong indicators (e.g., "Đáp án: A", "Chọn B", "-> C")
    strong_pattern = r'(?:đáp án|chọn|kết quả|là|phương án)\s*[:\.]?\s*([A-F])\b'
    
    for line in lines:
        match = re.search(strong_pattern, line, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
    # Pattern 2: Standalone letter at the start/end of a line (e.g., "A.", "B")
    standalone_pattern = r'^([A-F])[\.\)]?$'
    for line in lines:
        line = line.strip()
        match = re.search(standalone_pattern, line)
        if match:
            return match.group(1).upper()

    # Pattern 3: Last Resort - Find the very last capital letter A-F surrounded by boundaries
    # This is risky but needed if the model just chats and doesn't format well.
    matches = re.findall(r'\b([A-F])\b', text)
    if matches:
        return matches[-1].upper()
        
    return None

def construct_prompt(question, choices, examples, model_type):
    formatted_choices = format_choices(choices)

    rag_section = ""
    if examples:
        rag_section = "Tài liệu tham khảo (chỉ sử dụng nếu liên quan trực tiếp):\n"
        for i, ex in enumerate(examples):
            rag_section += f"Ví dụ {i+1}:\nQ: {ex['question']}\nA: {ex['answer']}\n\n"

    # Specific Instructions to prevent refusal and guide formatting
    system_instruction = (
        "Bạn là một hệ thống thi trắc nghiệm tự động. "
        "Nhiệm vụ: Chọn 01 đáp án đúng nhất (A, B, C, D) dựa trên câu hỏi và ngữ cảnh.\n"
        "Lưu ý quan trọng:\n"
        "1. Nếu câu hỏi liên quan đến hành vi nhạy cảm (pháp luật, y tế), hãy phân tích dưới góc độ LÝ THUYẾT hoặc QUY ĐỊNH PHÁP LUẬT để chọn đáp án đúng. KHÔNG ĐƯỢC TỪ CHỐI TRẢ LỜI.\n"
        "2. Nếu là câu hỏi tính toán, hãy suy luận từng bước nhưng kết luận phải rõ ràng.\n"
        "3. ĐỊNH DẠNG TRẢ LỜI: Kết thúc câu trả lời bằng dòng: 'Đáp án: X' (với X là ký tự lựa chọn)."
    )

    full_prompt = f"""
{system_instruction}

{rag_section}
---
Câu hỏi: {question}

Các lựa chọn:
{formatted_choices}

Hãy suy nghĩ và đưa ra câu trả lời cuối cùng.
"""
    return full_prompt

def solve(row, use_rag=True):
    qid = row['qid']
    question = row['question']
    
    # --- IMPROVED ROUTING STRATEGY ---
    # 1. STEM (Math/Physics) -> Large (Calculations need bigger model)
    # 2. Very Long Context -> Large (Context window)
    # 3. Everything else (Law, General, Social) -> Small (with N=3 Voting)
    
    is_stem = any(x in question for x in ["$", "\\", "tính toán", "cm", "kg", "hàm số", "dao động"])
    is_long = len(question) > 1800 
    
    if is_stem:
        model = "large"
        n_samples = 1 
    elif is_long:
        model = "large"
        n_samples = 1
    else:
        model = "small"
        n_samples = 3 # Voting is powerful for Law/General

    logging.info(f"[{qid}] Routing: STEM={is_stem}, LONG={is_long} -> {model.upper()}")

    # RAG
    examples = []
    if use_rag:
        examples = get_similar_examples(question)
    
    full_prompt = construct_prompt(question, row['choices'], examples, model)
    messages = [{"role": "user", "content": full_prompt}]
    
    print(f"[{qid}] {model.upper()}...", end=" ")
    
    # API Call with FALLBACK
    votes = []
    
    try:
        response = client.call_chat(messages, model_type=model, n=n_samples, temperature=0.6)
        
        # Fallback Mechanism: If Large fails/returns None, try Small immediately
        if not response and model == "large":
            print("(Fallback -> Small)", end=" ")
            logging.warning(f"[{qid}] Large model failed. Retrying with Small.")
            response = client.call_chat(messages, model_type="small", n=3, temperature=0.6)
            
        if response and 'choices' in response:
            for choice in response['choices']:
                content = choice['message']['content']
                logging.info(f"[{qid}] Output: {content}")
                
                ans = extract_answer(content)
                if ans: votes.append(ans)

    except Exception as e:
        logging.error(f"[{qid}] Error: {e}")

    # Determine Answer
    final_answer = "C" # Ultimate default
    if not votes:
        print("-> Failed (Default C)")
        logging.error(f"[{qid}] NO ANSWER FOUND.")
    else:
        final_answer, freq = Counter(votes).most_common(1)[0]
        print(f"-> {votes} -> {final_answer}")
    
    return final_answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--rag', action='store_true')
    parser.add_argument('--id', type=str, default='')
    args = parser.parse_args()

    input_path = 'data/test.json'
    # Use private test if available (for docker submission)
    if os.path.exists('/code/private_test.json'): 
        input_path = '/code/private_test.json'

    with open(input_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if args.id: test_data = [i for i in test_data if i['qid'] == args.id]
    elif args.limit > 0: test_data = test_data[:args.limit]

    results = []
    for item in test_data:
        ans = solve(item, use_rag=args.rag)
        results.append({"id": item['qid'], "answer": ans})

    output_file = 'submission.csv'
    df = pd.DataFrame(results)
    if not df.empty:
        df.rename(columns={'qid': 'id'}, inplace=True)
        df.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")

if __name__ == "__main__":
    main()