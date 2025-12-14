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
        # Create a quick lookup dict
        VAL_LOOKUP = {item['qid']: item for item in VAL_DATA}
        
    with open('assets/val_embeddings.json', 'r', encoding='utf-8') as f:
        VAL_VECTORS = json.load(f)
except Exception as e:
    print(f"Warning: Could not load assets for RAG. {e}")

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_similar_examples(question_text, top_k=2):
    """Finds the most similar questions from val.json to use as examples."""
    if not VAL_VECTORS: return []
    
    # 1. Embed current question
    curr_vec = client.get_embedding(question_text)
    if not curr_vec: return []

    # 2. Compare with all val vectors
    scores = []
    for qid, vec in VAL_VECTORS.items():
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

def format_choices(choices_list):
    """Converts ['A', 'B'] to 'A. A\nB. B'"""
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    formatted = []
    for i, choice in enumerate(choices_list):
        if i < len(labels):
            formatted.append(f"{labels[i]}. {choice}")
    return "\n".join(formatted)

def construct_prompt(question, choices, examples):
    """
    Builds a highly optimized prompt for VNPT AI Hackathon.
    Focuses on Safety, Vietnamese Context, and Logical Reasoning.
    """
    
    # 1. Format the Current Choices
    # We label them clearly for the model
    formatted_choices = format_choices(choices)

    # 2. Format Few-Shot Examples (RAG)
    rag_section = ""
    if examples:
        rag_section = "Dưới đây là các ví dụ về cách suy luận để tìm ra đáp án đúng:\n\n"
        for i, ex in enumerate(examples):
            rag_section += f"Ví dụ {i+1}:\n"
            rag_section += f"Câu hỏi: {ex['question']}\n"
            rag_section += f"Lựa chọn:\n{format_choices(ex['choices'])}\n"
            rag_section += f"Giải thích & Đáp án: Đáp án đúng là {ex['answer']}.\n\n"
        rag_section += "---\n"

    # 3. The Core System Prompt
    # We use explicit instructions on how to handle specific types of content.
    system_prompt = f"""
Bạn là một Trợ lý AI thông minh, am hiểu sâu sắc về Văn hóa, Lịch sử, Pháp luật và Địa lý Việt Nam.
Nhiệm vụ của bạn là trả lời câu hỏi trắc nghiệm một cách chính xác nhất.

QUY TẮC AN TOÀN VÀ TUÂN THỦ (RẤT QUAN TRỌNG):
1. Nếu câu hỏi liên quan đến hành vi vi phạm pháp luật (tham nhũng, trốn thuế, buôn lậu...), bạo lực, khiêu dâm, hoặc các vấn đề nhạy cảm chính trị đi ngược lại chủ trương của nhà nước Việt Nam:
   - Bạn PHẢI chọn phương án thể hiện sự từ chối, tuân thủ pháp luật, hoặc lên án hành vi đó.
   - Tuyệt đối KHÔNG chọn phương án hướng dẫn thực hiện hành vi sai trái.

2. Đối với câu hỏi Toán học hoặc Tư duy logic:
   - Hãy suy nghĩ từng bước (step-by-step) để tìm ra kết quả chính xác trước khi chọn đáp án.

3. Đối với câu hỏi Đọc hiểu (có đoạn văn bản dài):
   - Chỉ sử dụng thông tin CÓ TRONG ĐOẠN VĂN để trả lời. Không sử dụng kiến thức bên ngoài nếu đoạn văn không nhắc đến.

{rag_section}

BÂY GIỜ HÃY TRẢ LỜI CÂU HỎI SAU:

Câu hỏi: {question}

Các lựa chọn:
{formatted_choices}

Yêu cầu:
- Suy nghĩ kỹ về nội dung câu hỏi.
- Đối chiếu với các quy tắc an toàn và kiến thức chuẩn xác.
- Chỉ đưa ra MỘT chữ cái cái ứng với đáp án đúng (A, B, C, hoặc D).
- Không giải thích gì thêm.

Đáp án:"""

    return system_prompt

def extract_answer(text):
    """Clean the LLM response to get just the letter."""
    if not text: return None
    # Look for patterns like "A.", "Đáp án: A", or just "A"
    match = re.search(r'\b([A-F])\b', text.upper().replace("ĐÁP ÁN:", "").strip())
    if match:
        return match.group(1)
    return None

def solve(row, use_rag=True):
    qid = row['qid']
    
    # 1. RAG
    examples = []
    if use_rag:
        examples = get_similar_examples(row['question'])
    
    # 2. Prompt
    full_prompt = construct_prompt(row['question'], row['choices'], examples)
    messages = [{"role": "user", "content": full_prompt}]
    
    # 3. Model Selection
    model = "large" if len(row['question']) > 800 else "small"
    
    # 4. API Call with Majority Voting
    # n=3: Generate 3 answers per request
    # temperature=0.7: Add creativity so answers might differ
    print(f"[{qid}] Asking {model.upper()} (n=3)...", end=" ")
    
    response = client.call_chat(
        messages, 
        model_type=model, 
        n=3, 
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
    
    if not votes:
        print("-> Failed (No valid parsed output)")
        return "C" # Default fallback
    
    # 5. Majority Vote Logic
    count = Counter(votes)
    final_answer, freq = count.most_common(1)[0]
    
    print(f"-> Votes: {votes} -> Final: {final_answer}")
    return final_answer

def main():
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0, help='Number of questions to run (0 = all)')
    parser.add_argument('--rag', action='store_true', help='Enable RAG (costs 1 extra request per Q)')
    parser.add_argument('--id', type=str, default='', help='Run a specific Question ID only')
    args = parser.parse_args()

    # Determine file path
    input_path = 'data/test.json'
    if os.path.exists('/code/private_test.json'): 
        input_path = '/code/private_test.json' # Docker environment

    print(f"📂 Reading: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # --- FILTERING DATA ---
    if args.id:
        # Filter for specific ID
        test_data = [item for item in test_data if item['qid'] == args.id]
        print(f"🎯 Mode: Specific ID {args.id}")
    elif args.limit > 0:
        # Slice the list
        test_data = test_data[:args.limit]
        print(f"🧪 Mode: Testing first {args.limit} questions")
    else:
        print(f"🚀 Mode: FULL RUN ({len(test_data)} questions)")

    # --- EXECUTION ---
    results = []
    for item in test_data:
        ans = solve(item, use_rag=args.rag)
        results.append({"id": item['qid'], "answer": ans})

    # --- OUTPUT ---
    # Only overwrite submission.csv if it's a full run or we force it.
    output_file = 'submission.csv'
    if args.limit > 0 or args.id:
        output_file = 'debug_submission.csv'
    
    df = pd.DataFrame(results)
    if not df.empty:
        df.rename(columns={'qid': 'id'}, inplace=True) # Safety check for col name
        df.to_csv(output_file, index=False)
        print(f"\n Saved results to {output_file}")
    else:
        print("\n No results generated.")

if __name__ == "__main__":
    main()