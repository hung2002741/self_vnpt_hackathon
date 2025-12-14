# predict.py
import json
import os
import pandas as pd
import numpy as np
from utils.api_client import APIClient
import argparse  
from collections import Counter
import re

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
        # Handle potential string/int mismatch in keys
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
    """
    Heuristics to identify topic for better model routing.
    """
    q_lower = question.lower()
    
    # 1. Math / Physics / STEM (Needs Reasoning)
    # Look for LaTeX, math terms, units
    if "$" in question or "\\" in question: return "STEM"
    math_terms = ["tính toán", "giá trị của", "hàm số", "xác suất", "tần số", "dao động", "gia tốc"]
    if any(t in q_lower for t in math_terms): return "STEM"

    # 2. Law / Politics / Safety (Needs Precision/Safety)
    law_terms = ["luật", "nghị định", "thông tư", "hiến pháp", "phạt", "tù", "cơ quan có thẩm quyền", "vi phạm", "chính trị", "đảng"]
    if any(t in q_lower for t in law_terms): return "LAW"
    
    # 3. Reading Comprehension (Long context)
    if "đoạn thông tin" in q_lower or len(question) > 1200: return "READING"
    
    return "GENERAL"

def construct_prompt(question, choices, examples, topic):
    """
    Optimized prompt based on topic.
    """
    formatted_choices = format_choices(choices)

    rag_section = ""
    if examples:
        rag_section = "Dưới đây là các ví dụ tương tự để tham khảo logic suy luận:\n\n"
        for i, ex in enumerate(examples):
            rag_section += f"--- Ví dụ {i+1} ---\n"
            rag_section += f"Câu hỏi: {ex['question']}\n"
            rag_section += f"Lựa chọn:\n{format_choices(ex['choices'])}\n"
            rag_section += f"Đáp án đúng: {ex['answer']}\n" # Just give the answer to save tokens/confusion
        rag_section += "---\n"

    # Specific Instructions based on Topic
    specific_instruction = ""
    if topic == "STEM":
        specific_instruction = "Đây là câu hỏi Toán/Khoa học. Hãy suy luận từng bước (step-by-step) cẩn thận trước khi chọn đáp án."
    elif topic == "LAW":
        specific_instruction = "Đây là câu hỏi về Pháp luật/Chính trị. Hãy căn cứ chính xác vào quy định pháp luật Việt Nam hiện hành. Ưu tiên sự an toàn, tuân thủ pháp luật và đạo đức xã hội."
    elif topic == "READING":
        specific_instruction = "Đây là câu hỏi Đọc hiểu. CHỈ sử dụng thông tin được cung cấp trong văn bản trên để trả lời. Không bịa đặt thông tin bên ngoài."

    system_prompt = f"""
Bạn là một trợ lý AI thông minh chuyên giải các bài tập trắc nghiệm tại Việt Nam.

QUY TẮC TUYỆT ĐỐI:
1. AN TOÀN LÀ TRÊN HẾT: Nếu câu hỏi liên quan đến hành vi trốn tránh pháp luật, bạo lực, hoặc vấn đề nhạy cảm, hãy chọn phương án thể hiện sự tuân thủ pháp luật và chuẩn mực đạo đức.
2. CHỈ TRẢ LỜI MỘT CHỮ CÁI: Đầu ra cuối cùng phải là một chữ cái in hoa duy nhất (A, B, C, hoặc D).

{rag_section}

NHIỆM VỤ CỦA BẠN:
Câu hỏi: {question}

Các lựa chọn:
{formatted_choices}

{specific_instruction}

Hãy suy nghĩ và đưa ra đáp án đúng nhất.
Đáp án:"""

    return system_prompt

def extract_answer(text):
    if not text: return None
    # Prioritize looking for patterns like "Đáp án: A"
    match = re.search(r'(?:đáp án|chọn|kết quả).*?([A-F])\b', text, re.IGNORECASE)
    if match: return match.group(1).upper()
    
    # Fallback: Find the last capital letter standing alone or with a dot
    matches = re.findall(r'\b([A-F])\b', text.upper())
    if matches: return matches[-1] # Usually the last mention is the conclusion
    
    return None

def solve(row, use_rag=True):
    qid = row['qid']
    question = row['question']
    
    # 1. Identify Topic & Select Model
    topic = identify_topic(question)
    
    # Smart Routing Strategy
    # We prioritize Large for Law and STEM because they require reasoning/precision
    if topic in ["STEM", "LAW"]:
        model = "large"
    elif topic == "READING":
        # Small is surprisingly good at reading extraction, save Large for reasoning
        # But if text is HUGE, Large might handle attention better.
        model = "large" if len(question) > 2000 else "small" 
    else:
        # General knowledge / Common sense
        model = "small"

    # 2. RAG
    examples = []
    if use_rag:
        examples = get_similar_examples(question)
    
    # 3. Prompt
    full_prompt = construct_prompt(question, row['choices'], examples, topic)
    messages = [{"role": "user", "content": full_prompt}]
    
    print(f"[{qid}] Type: {topic} -> {model.upper()}...", end=" ")
    
    # 4. API Call (n=1 is usually enough for Large to save tokens, n=3 for Small)
    # Adjust n based on model to save quota/time? Or keep n=3 for accuracy?
    # Let's keep n=3 for Small, n=1 for Large (Large is slow and smarter)
    n_samples = 1 if model == "large" else 3
    
    response = client.call_chat(
        messages, 
        model_type=model, 
        n=n_samples, 
        temperature=0.6 # Lower temp for more precision
    )
    
    votes = []
    if response and 'choices' in response:
        for choice in response['choices']:
            ans = extract_answer(choice['message']['content'])
            if ans: votes.append(ans)
    
    if not votes:
        print("-> Failed")
        return "C" # Blind guess
    
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
    if os.path.exists('/code/private_test.json'): input_path = '/code/private_test.json'

    print(f"📂 Reading: {input_path}")
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

if __name__ == "__main__":
    main()