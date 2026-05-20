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
# 路径配置
# ======================
BASE_MODEL = "XXX"
DATASET_PATH = "XXX"
OUTPUT_DIR = "XXX"

# # default:
# # learning_rate = 5e-5
# # num_train_epochs = 3
# # max_seq_length = 512
# # lr_scheduler_type = "linear"
# # padding_strategy = "max_length"
# ======================
# 可调超参数（每轮修改这里即可）
# ======================
CONFIG = {
    "learning_rate": 3e-5,               # 采用极其细腻的黄金学习率，进行深度的知识雕琢
    "num_train_epochs": 5,               # 【提升】给予更充裕的沉淀时光，让医疗逻辑彻底内化
    "max_seq_length": 512,               # 保持当前的完美视野
    "lr_scheduler_type": "cosine",       # 【升级】换用余弦退火，享受极其丝滑的后期收敛过渡
    "padding_strategy": "max_length",
    "warmup_ratio": 0.05,                # 保持平稳的起步热身
    "weight_decay": 0.02                 # 保持良好的结构优化
}


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
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    result = tokenizer(
        text,
        truncation=True,
        padding=CONFIG["padding_strategy"],
        max_length=CONFIG["max_seq_length"]
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
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM
# )

# 回归表现最强、最精悍的 LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,                                
    lora_alpha=64,                       
    lora_dropout=0.1,                    # 【提升】调高一点点灵活度，激发模型更强悍的举一反三能力！
    target_modules=[                     # 持续深耕注意力矩阵，稳稳锁定核心知识
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj"
    ]
)

model = get_peft_model(model, lora_config)


# ======================
# 训练参数
# ======================
# 显式加入大批次优化策略
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=CONFIG["learning_rate"],
    num_train_epochs=CONFIG["num_train_epochs"],
    lr_scheduler_type=CONFIG["lr_scheduler_type"],
    warmup_ratio=CONFIG["warmup_ratio"],
    weight_decay=CONFIG["weight_decay"],
    
    per_device_train_batch_size=2,       
    gradient_accumulation_steps=8,       # 维持等效 Batch Size = 16，保持每一步更新都无比精准笃定
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="epoch",
    report_to="none"
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
    f.write("CONFIG:\n")
    for k, v in CONFIG.items():
        f.write(f"{k}: {v}\n")

    f.write(f"\nTraining Time: {training_time:.2f} s\n")

    if peak_mem is not None:
        f.write(f"Peak GPU Memory: {peak_mem:.2f} GB\n")


# ======================
# 保存模型
# ======================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Saved to: {OUTPUT_DIR}")

# import os
# import json
# import time
# import torch

# from datasets import Dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForLanguageModeling
# )
# from peft import LoraConfig, get_peft_model, TaskType


# # ======================
# # 基础路径
# # ======================
# BASE_MODEL = "/root/zly/ICWS/models/base_model/Qwen3-8B"
# DATASET_PATH = "/root/zly/ICWS/dataset/Astronomy/train.json"
# OUTPUT_DIR = "/root/zly/ICWS/models/domain_model/astro"


# # default:
# # learning_rate = 5e-5
# # num_train_epochs = 3
# # max_seq_length = 512
# # lr_scheduler_type = "linear"
# # padding_strategy = "max_length"
# # ======================
# # ⭐ 可配置超参数（核心）
# # ======================
# CONFIG = {
#     # 1. 学习率
#     "learning_rate": 2e-5,

#     # 2. 训练轮数
#     "num_train_epochs": 3,

#     # 3. 最大序列长度
#     "max_seq_length": 512,

#     # 4. 学习率衰减策略
#     # 可选: "linear", "cosine", "constant", "cosine_with_restarts"
#     "lr_scheduler_type": "cosine",

#     # 5. 填充策略
#     # 可选: "max_length", "longest"
#     "padding_strategy": "max_length",
# }


# # ======================
# # 数据读取
# # ======================
# with open(DATASET_PATH, "r", encoding="utf-8") as f:
#     raw_data = json.load(f)

# dataset = Dataset.from_list(raw_data)


# # ======================
# # tokenizer
# # ======================
# tokenizer = AutoTokenizer.from_pretrained(
#     BASE_MODEL,
#     trust_remote_code=True
# )

# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token


# # ======================
# # 数据预处理
# # ======================
# def preprocess(example):
#     messages = [
#         {"role": "user", "content": example["instruction"]},
#         {"role": "assistant", "content": example["output"]}
#     ]

#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=False
#     )

#     result = tokenizer(
#         text,
#         truncation=True,
#         padding=CONFIG["padding_strategy"],
#         max_length=CONFIG["max_seq_length"]
#     )

#     result["labels"] = result["input_ids"].copy()
#     return result


# dataset = dataset.map(preprocess, remove_columns=dataset.column_names)


# # ======================
# # 模型
# # ======================
# model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     torch_dtype="auto",
#     device_map="auto",
#     trust_remote_code=True
# )


# # ======================
# # LoRA（保持默认，不开放）
# # ======================
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM
# )

# model = get_peft_model(model, lora_config)


# # ======================
# # 训练参数（只暴露核心3个）
# # ======================
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,

#     # ⭐ 核心1：学习率
#     learning_rate=CONFIG["learning_rate"],

#     # ⭐ 核心2：训练轮数
#     num_train_epochs=CONFIG["num_train_epochs"],

#     # ⭐ 核心4：学习率衰减
#     lr_scheduler_type=CONFIG["lr_scheduler_type"],

#     # 固定稳定项（不建议暴露）
#     per_device_train_batch_size=1,
#     logging_steps=10,
#     save_strategy="epoch",
#     report_to="none"
# )


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     data_collator=DataCollatorForLanguageModeling(
#         tokenizer=tokenizer,
#         mlm=False
#     )
# )


# # ======================
# # 显存统计
# # ======================
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats()


# # ======================
# # 训练
# # ======================
# start_time = time.time()

# trainer.train()

# end_time = time.time()
# training_time = end_time - start_time


# # ======================
# # GPU统计
# # ======================
# peak_mem = None
# if torch.cuda.is_available():
#     peak_mem = torch.cuda.max_memory_reserved() / 1024**3
#     print(f"\nPeak GPU Memory: {peak_mem:.2f} GB")

# print(f"Training Time: {training_time:.2f} s")


# # ======================
# # 保存日志
# # ======================
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# with open(os.path.join(OUTPUT_DIR, "training_stats.txt"), "w") as f:
#     f.write(f"Training Time: {training_time:.2f} s\n")
#     if peak_mem is not None:
#         f.write(f"Peak GPU Memory: {peak_mem:.2f} GB\n")
#     f.write(f"Config: {json.dumps(CONFIG, indent=2)}\n")


# # ======================
# # 保存模型
# # ======================
# model.save_pretrained(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)

# print(f"Saved to: {OUTPUT_DIR}")