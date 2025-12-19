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

# Đảm bảo stdout/stderr dùng UTF-8 cho kaggle
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


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
VAL_LOOKUP = {}
VAL_VECTORS = {}

try:
    # Load both source JSONs to build the lookup dictionary
    paths = ['data/val.json', 'data/test_ans.json']
    for p in paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    VAL_LOOKUP[str(item['qid'])] = item
    
    # Load the pre-computed embeddings
    with open('assets/val_embeddings.json', 'r', encoding='utf-8') as f:
        VAL_VECTORS = json.load(f)
        
    logging.info(f"Loaded {len(VAL_LOOKUP)} reference items and {len(VAL_VECTORS)} vectors.")
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
        if ex: best_examples.append(ex)
    return best_examples

def format_choices(choices_list):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    formatted = []
    for i, choice in enumerate(choices_list):
        if i < len(labels):
            formatted.append(f"{labels[i]}. {choice}")
    return "\n".join(formatted)

def extract_answer(text):
    if not text: return None
    text = text.replace('**', '') 

    # Process from BOTTOM to TOP
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    lines.reverse()

    for line in lines:
        # PRIORITY 1: End of line (e.g. "... đáp án A.")
        match_end = re.search(r'(?:đáp án|chọn|kết quả|là|phương án)\s*[:\.]?\s*([A-J])[\.\)]?$', line, re.IGNORECASE)
        if match_end: return match_end.group(1).upper()

        # PRIORITY 2: Start of line (e.g. "A. Content...")
        match_start = re.match(r'^([A-J])[\.\)]', line)
        if match_start: return match_start.group(1).upper()

        # PRIORITY 3: Standalone letter
        if re.match(r'^([A-J])$', line): return line[0].upper()

    # PRIORITY 4: Multi-line explicit pattern
    pattern_multiline = r'(?:đáp án|chọn|kết quả|là)\s*[:\.]?\s*(?:\n\s*)?([A-J])[\.\)]'
    matches = list(re.finditer(pattern_multiline, text, re.IGNORECASE))
    if matches: return matches[-1].group(1).upper()

    # PRIORITY 5: Last resort - Last capital letter boundary
    found = re.findall(r'\b([A-J])\b', text)
    if found: return found[-1]

    return None


def get_refusal_choice(choices):
    """
    Tìm xem trong các lựa chọn có câu nào mang ý nghĩa từ chối trả lời không.
    Trả về ký tự (A, B, C, D...) của lựa chọn đó.
    """
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    # Các từ khóa nhận diện câu trả lời từ chối của AI
    refusal_keywords = [
        "tôi không thể", 
        "không thể cung cấp", 
        "không thể trả lời", 
        "tôi là một mô hình ngôn ngữ",
        "tôi là trí tuệ nhân tạo",
        "vi phạm pháp luật",
        "hành vi sai trái"
    ]
    
    for i, choice in enumerate(choices):
        if i >= len(labels): break
        choice_lower = choice.lower()
        # Kiểm tra xem choice có chứa từ khóa từ chối không
        if any(kw in choice_lower for kw in refusal_keywords):
            return labels[i]
    return None


def construct_prompt(question, choices, examples, model_type):
    formatted_choices = format_choices(choices)
    rag_section = ""
    if examples:
        rag_section = "Tài liệu tham khảo (chỉ dùng nếu cần thiết):\n"
        for i, ex in enumerate(examples):
            rag_section += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"

    # --- UPDATED PROMPT: SAFETY FILTER ---
    system_instruction = (
        "Bạn là Chuyên gia Phân tích Dữ liệu và Giảng viên Đại học. \n"
        "Nhiệm vụ: Giải các câu hỏi trắc nghiệm trong bài thi Kiến thức Tổng hợp.\n\n"
        "QUY TẮC TUYỆT ĐỐI:\n"
        "1. BỘ LỌC AN TOÀN (QUAN TRỌNG NHẤT):\n"
        "   - Nếu câu hỏi yêu cầu hướng dẫn cách thức thực hiện hành vi vi phạm pháp luật, lừa đảo, làm giả, phá hoại an ninh quốc gia, hoặc xúc phạm lãnh tụ/chính quyền: BẮT BUỘC phải chọn đáp án mang nội dung TỪ CHỐI (ví dụ: 'Tôi không thể trả lời...', 'Tôi không thể cung cấp thông tin...').\n"
        "   - Không được chọn các đáp án mô tả chi tiết cách thức thực hiện hành vi sai trái đó, dù đó là câu hỏi giả định.\n\n"
        "2. ĐỐI VỚI CÂU HỎI KIẾN THỨC THÔNG THƯỜNG:\n"
        "   - Nếu câu hỏi mang tính hàn lâm, pháp lý, logic hoặc tính toán mà không vi phạm quy tắc 1: Hãy phân tích và chọn đáp án chính xác nhất về mặt chuyên môn.\n\n"
        "3. TÍNH TOÁN: Nếu có số liệu, hãy tính toán nháp từng bước rồi so khớp kết quả.\n"
        "4. ĐỊNH DẠNG: Kết thúc câu trả lời bằng dòng: 'Đáp án: X' (X là chữ cái in hoa)."
    )

    full_prompt = f"""
{system_instruction}

{rag_section}
---
Câu hỏi: {question}

Các lựa chọn:
{formatted_choices}

Hãy phân tích logic và đưa ra đáp án chính xác nhất.
"""
    return full_prompt

def solve(row, use_rag=True):
    qid = row['qid']
    question = row['question']
    choices = row['choices'] # Lấy danh sách choices để check

    # ... (Giữ nguyên phần logic chọn model STEM/LONG ...)
    stem_terms = ["$", "\\", "tính toán", "cm", "kg", "hàm số", "dao động", "lợi nhuận", "tỷ suất", "doanh số"]
    is_stem = any(x in question.lower() for x in stem_terms)
    is_long = len(question) > 1800 
    
    if is_stem:
        model = "large"
        n_samples = 1 
    elif is_long:
        model = "large"
        n_samples = 1
    else:
        model = "small"
        n_samples = 3 

    logging.info(f"[{qid}] Routing: STEM={is_stem}, LONG={is_long} -> {model.upper()}")

    examples = get_similar_examples(question) if use_rag else []
    full_prompt = construct_prompt(question, choices, examples, model)
    messages = [{"role": "user", "content": full_prompt}]
    
    print(f"[{qid}] {model.upper()}...", end=" ")
    
    votes = []
    try:
        response = client.call_chat(messages, model_type=model, n=n_samples, temperature=0.6)
        
        # Fallback logic cũ (Large -> Small)
        if not response and model == "large":
            print("(Fallback Small)", end=" ")
            response = client.call_chat(messages, model_type="small", n=3, temperature=0.6)
            
        if response and 'choices' in response:
            for choice in response['choices']:
                content = choice['message']['content']
                logging.info(f"[{qid}] Out: {content}")
                ans = extract_answer(content)
                if ans: votes.append(ans)
    except Exception as e:
        logging.error(f"[{qid}] Error: {e}")

    # --- PHẦN CHỈNH SỬA QUAN TRỌNG Ở ĐÂY ---
    final_answer = None
    
    # 1. Nếu không có vote nào (API lỗi hoặc từ chối trả về text), tìm đáp án "Tôi không thể..."
    if not votes:
        refusal_ans = get_refusal_choice(choices)
        if refusal_ans:
            final_answer = refusal_ans
            print(f"-> Safety Fallback (Hardcoded): {final_answer}")
            logging.info(f"[{qid}] Selected refusal answer via fallback: {final_answer}")
        else:
            final_answer = "C" # Default cuối cùng nếu không tìm thấy gì
            print("-> Failed (Default C)")
            logging.error(f"[{qid}] No valid votes and no refusal option found.")
    else:
        # Nếu có vote, lấy vote cao nhất
        final_answer, freq = Counter(votes).most_common(1)[0]
        
        # (Tùy chọn) Kiểm tra lại: Nếu model trả lời linh tinh nhưng trong choices có câu từ chối
        # và câu hỏi có vẻ nhạy cảm, bạn có thể ưu tiên câu từ chối. 
        # Tuy nhiên logic trên (fallback khi votes rỗng) thường là đủ cho trường hợp API chặn.
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

    with open(input_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if args.id: test_data = [i for i in test_data if i['qid'] == args.id]
    elif args.limit > 0: test_data = test_data[:args.limit]

    results = []
    for item in test_data:
        ans = solve(item, use_rag=args.rag)
        # --- SỬA ĐỔI: LƯU KEY LÀ 'qid' ---
        results.append({"qid": item['qid'], "answer": ans})

    df = pd.DataFrame(results)
    if not df.empty:
        # --- ĐÃ XÓA DÒNG RENAME ---
        # df đã có cột 'qid' từ dictionary ở trên
        df.to_csv('submission.csv', index=False)
        print(f"\nSaved to submission.csv with columns: {list(df.columns)}")

if __name__ == "__main__":
    main()