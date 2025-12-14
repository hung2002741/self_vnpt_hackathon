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
# This creates a file 'execution_debug.log' that records every step
logging.basicConfig(
    filename='execution_debug.log', 
    filemode='w', # Overwrite file on each new run
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

# Initialize Client
client = APIClient()

# Load Assets (Global)
VAL_DATA = []
VAL_VECTORS = {}
try:
    with open('data/val.json', 'r', encoding='utf-8') as f:
        VAL_DATA = json.load(f)
        VAL_LOOKUP = {str(item['qid']): item for item in VAL_DATA}
        
    with open('assets/val_embeddings.json', 'r', encoding='utf-8') as f:
        VAL_VECTORS = json.load(f)
except Exception as e:
    print(f"Warning: Could not load assets for RAG. {e}")
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

def identify_topic(question):
    q_lower = question.lower()
    
    # PRIORITY 1: Reading Comprehension (Context Extraction)
    # If the user provides a text block, we MUST use it. 
    # This fixes test_0077 being treated as "Law" just because it mentions "government".
    if "đoạn thông tin" in q_lower or len(question) > 1000: 
        return "READING"

    # PRIORITY 2: STEM (Math/Science requires specific reasoning)
    if "$" in question or "\\" in question: return "STEM"
    math_terms = ["tính toán", "giá trị của", "hàm số", "xác suất", "tần số", "dao động", "gia tốc", "cm", "kg"]
    if any(t in q_lower for t in math_terms): return "STEM"

    # PRIORITY 3: Law / Politics (External Knowledge required)
    law_terms = ["luật", "nghị định", "thông tư", "hiến pháp", "phạt", "tù", "cơ quan", "vi phạm", "chính trị", "đảng", "bộ luật"]
    if any(t in q_lower for t in law_terms): return "LAW"
    
    return "GENERAL"

def construct_prompt(question, choices, examples, topic):
    formatted_choices = format_choices(choices)

    rag_section = ""
    if examples:
        rag_section = "Dưới đây là các ví dụ tham khảo:\n\n"
        for i, ex in enumerate(examples):
            rag_section += f"Ví dụ {i+1}:\nCâu hỏi: {ex['question']}\nĐáp án đúng: {ex['answer']}\n\n"
        rag_section += "---\n"

    # Specific Instructions
    specific_instruction = ""
    if topic == "STEM":
        specific_instruction = "Đây là câu hỏi Toán/Khoa học. Hãy suy luận từng bước (step-by-step) để tìm kết quả chính xác."
    elif topic == "LAW":
        specific_instruction = "Đây là câu hỏi về Pháp luật/Chính trị. Hãy căn cứ vào các văn bản pháp luật hiện hành và chuẩn mực đạo đức."
    elif topic == "READING":
        # Crucial change for History/Biography questions:
        specific_instruction = "Đây là câu hỏi Đọc hiểu và trích xuất thông tin. Hãy TRẢ LỜI DỰA TRÊN ĐOẠN VĂN ĐƯỢC CUNG CẤP. Tuyệt đối trung thực với nội dung đoạn văn."

    system_prompt = f"""
Bạn là trợ lý AI thông minh.

QUY TẮC:
1. Trả lời MỘT chữ cái in hoa duy nhất (A, B, C, D).
2. Nếu câu hỏi yêu cầu Đọc hiểu, hãy chỉ dùng thông tin trong văn bản để trả lời.

{rag_section}

Câu hỏi: {question}

Lựa chọn:
{formatted_choices}

{specific_instruction}

Đáp án:"""
    return system_prompt

def extract_answer(text):
    if not text: return None
    match = re.search(r'(?:đáp án|chọn|kết quả).*?([A-F])\b', text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    matches = re.findall(r'\b([A-F])\b', text.upper())
    if matches: return matches[-1]
    return None

def solve(row, use_rag=True):
    qid = row['qid']
    question = row['question']
    
    # 1. Routing
    topic = identify_topic(question)
    
    if topic in ["STEM", "LAW"]:
        model = "large"
    elif topic == "READING":
        model = "large" if len(question) > 2000 else "small" 
    else:
        model = "small"

    logging.info(f"[{qid}] PROCESSING | Topic: {topic} | Model: {model.upper()}")

    # 2. RAG
    examples = []
    if use_rag:
        examples = get_similar_examples(question)
    
    # 3. Prompt
    full_prompt = construct_prompt(question, row['choices'], examples, topic)
    messages = [{"role": "user", "content": full_prompt}]
    
    print(f"[{qid}] Type: {topic} -> {model.upper()}...", end=" ")
    
    # 4. API Call
    # n=1 for Large to avoid timeout/quota issues, n=3 for Small for accuracy
    n_samples = 1 if model == "large" else 3
    
    response = client.call_chat(
        messages, 
        model_type=model, 
        n=n_samples, 
        temperature=0.6
    )
    
    votes = []
    if response and 'choices' in response:
        for i, choice in enumerate(response['choices']):
            content = choice['message']['content']
            
            # --- LOG RAW OUTPUT ---
            logging.info(f"[{qid}] Raw Output #{i}: {content}")
            
            ans = extract_answer(content)
            if ans: 
                votes.append(ans)
                logging.info(f"[{qid}] Extracted: {ans}")
            else:
                logging.warning(f"[{qid}] extraction failed for: {content}")
    
    final_answer = "C" # Default
    if not votes:
        print("-> Failed")
        logging.error(f"[{qid}] FAILED - No valid votes obtained.")
        return "C"
    
    final_answer, freq = Counter(votes).most_common(1)[0]
    print(f"-> {votes} -> {final_answer}")
    logging.info(f"[{qid}] FINAL ANSWER: {final_answer}")
    
    return final_answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--rag', action='store_true')
    parser.add_argument('--id', type=str, default='')
    args = parser.parse_args()

    input_path = 'data/test.json'
    if os.path.exists('/code/private_test.json'): input_path = '/code/private_test.json'

    print(f"📂 Reading: {input_path}")
    logging.info(f"Starting run. Input: {input_path}, RAG: {args.rag}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if args.id: test_data = [i for i in test_data if i['qid'] == args.id]
    elif args.limit > 0: test_data = test_data[:args.limit]

    results = []
    for item in test_data:
        ans = solve(item, use_rag=args.rag)
        results.append({"id": item['qid'], "answer": ans})

    output_file = 'submission.csv'
    if args.limit > 0 or args.id: output_file = 'debug_submission.csv'
    
    df = pd.DataFrame(results)
    if not df.empty:
        df.rename(columns={'qid': 'id'}, inplace=True)
        df.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")
        logging.info("Run complete. Results saved.")

if __name__ == "__main__":
    main()