# 学习率：5e-05
# 训练轮数：3
# 学习率衰减：linear
# 最大序列长度：默认是不截断

import os
import json
import time
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType


# ======================
# 直接配置路径
# ======================
BASE_MODEL = "XXX"
DATASET_PATH = "XXX"
OUTPUT_DIR = "XXX"


# ======================
# 读取数据
# ======================
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data)


# ======================
# tokenizer
# ======================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ======================
# 数据预处理
# ======================
def preprocess(example):
    messages = [
        {
            "role": "user",
            "content": example["instruction"]
        },
        {
            "role": "assistant",
            "content": example["output"]
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    result = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    result["labels"] = result["input_ids"].copy()
    return result


dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names
)


# ======================
# 模型
# ======================
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)


# ======================
# LoRA
# ======================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)


# ======================
# 训练参数
# ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)


# ======================
# 显存统计初始化
# ======================
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ======================
# 开始训练（计时）
# ======================
start_time = time.time()

trainer.train()

end_time = time.time()
training_time = end_time - start_time


# ======================
# 输出统计信息
# ======================
peak_mem = None
if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_reserved() / 1024**3
    print(f"\nPeak GPU Memory: {peak_mem:.2f} GB")

print(f"Training Time: {training_time:.2f} s")


# ======================
# 保存统计信息
# ======================
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "training_stats.txt"), "w") as f:
    f.write(f"Training Time: {training_time:.2f} s\n")
    if peak_mem is not None:
        f.write(f"Peak GPU Memory: {peak_mem:.2f} GB\n")


# ======================
# 保存模型
# ======================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Saved to: {OUTPUT_DIR}")