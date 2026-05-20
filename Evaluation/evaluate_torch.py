import json
import re
import time
import pandas as pd
from tqdm import tqdm
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ======================
# 配置
# ======================
JSON_PATH = "XXX"
OUTPUT_XLSX = "XXX"
TEMP_SAVE_XLSX = "XXX"

BASE_MODEL_PATH = "XXX"
LORA_PATH = "XXX"

# LORA_PATH = None   # 如果评估LoRA模型，填路径；否则保持None

SAVE_INTERVAL = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 8

# ======================
# 加载模型
# ======================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True
)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 如果有LoRA，则加载
if LORA_PATH is not None:
    print(f"Loading LoRA from {LORA_PATH} ...")
    model = PeftModel.from_pretrained(model, LORA_PATH)

model.eval()


def extract_choice(text: str):
    text = text.strip().upper()

    match = re.search(r'\b([ABCD])\b', text)
    if match:
        return match.group(1)

    for ch in text:
        if ch in ['A', 'B', 'C', 'D']:
            return ch

    return "INVALID"


def ask_model(question_text: str):
    try:
        # 强制关闭思考，严格单字母输出
        prompt = (
            "You are a multiple-choice question answering assistant.\n"
            "Answer the following question.\n"
            "Only output one capital letter from A, B, C, or D.\n"
            "Do not explain.\n"
            "Do not think step by step.\n\n"
            f"{question_text}\n"
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(
            text,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        return response

    except Exception as e:
        print(f"\n调用失败: {e}")
        return "ERROR"


def save_excel(results, path, final_accuracy=None):
    df = pd.DataFrame(results)

    if final_accuracy is not None:
        summary = pd.DataFrame([{
            "question": "FINAL_ACCURACY",
            "model_raw_output": "",
            "model_answer": "",
            "gold_answer": "",
            "correct": final_accuracy
        }])
        df = pd.concat([df, summary], ignore_index=True)

    df.to_excel(path, index=False)


def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    correct_count = 0

    progress_bar = tqdm(
        enumerate(data, start=1),
        total=len(data),
        desc="Evaluating"
    )

    for idx, item in progress_bar:
        start_time = time.time()

        instruction = item["instruction"]
        gold = item["output"].strip().upper()

        raw_response = ask_model(instruction)
        pred = extract_choice(raw_response)

        is_correct = pred == gold
        if is_correct:
            correct_count += 1

        elapsed = time.time() - start_time
        current_acc = correct_count / idx

        results.append({
            "question": instruction,
            "model_raw_output": raw_response,
            "model_answer": pred,
            "gold_answer": gold,
            "correct": is_correct,
            "time_sec": round(elapsed, 3)
        })

        progress_bar.set_postfix({
            "acc": f"{current_acc:.4f}",
            "last": pred,
            "gold": gold,
            "time": f"{elapsed:.2f}s"
        })

        if idx % SAVE_INTERVAL == 0:
            save_excel(results, TEMP_SAVE_XLSX)
            print(f"\n已临时保存: {TEMP_SAVE_XLSX}")

    total = len(results)
    accuracy = correct_count / total if total > 0 else 0

    save_excel(results, OUTPUT_XLSX, final_accuracy=accuracy)

    print("\n" + "=" * 60)
    print(f"总题数: {total}")
    print(f"答对数量: {correct_count}")
    print(f"准确率: {accuracy:.4f}")
    print(f"最终结果保存至: {OUTPUT_XLSX}")
    print("=" * 60)

    os.remove(TEMP_SAVE_XLSX)


if __name__ == "__main__":
    total_start_time = time.perf_counter()
    main()
    total_end_time = time.perf_counter()
    total_elapsed = total_end_time - total_start_time
    print(f"总耗时: {total_elapsed:.2f} 秒")