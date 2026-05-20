import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_MODEL_PATH = "/home/zly/Qwen3-14B"
DEFAULT_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_VLLM_MODEL = os.environ.get("VLLM_MODEL", "")
DEFAULT_VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")
DEFAULT_KNOWLEDGE_GRAPH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tupu.txt")
MAX_KNOWLEDGE_GRAPH_CHARS = 12000

# 新增：Flag 和上轮记录文件的路径配置
DEFAULT_FLAG_PATH = "/root/zly/ICWS/flag.txt"
DEFAULT_TRAINING_LAST_PATH = "/root/zly/ICWS/traing_las.txt"

VECTOR_FIELDS = [
    "learning_rate",
    "epochs",
    "lr_scheduler",
    "finetune_method",
    "max_seq_length",
    "padding_strategy",
    "task_type",
    "sequence_type",
    "evaluation_metric",
]

LABELS_ZH = {
    "learning_rate": "学习率",
    "epochs": "训练轮数",
    "lr_scheduler": "学习率衰减",
    "finetune_method": "微调方法",
    "max_seq_length": "最大序列长度",
    "padding_strategy": "填充策略",
    "task_type": "任务类型",
    "sequence_type": "序列类型",
    "evaluation_metric": "最优模型评价指标",
}

ENCODERS = {
    "lr_scheduler": {"constant": 1, "linear": 2, "cosine": 3, "cosine_with_restarts": 4},
    "finetune_method": {"LoRA": 1, "QLoRA": 2, "Full fine-tuning": 3},
    "padding_strategy": {"max_length": 1, "longest": 2, "dynamic": 3},
    "task_type": {
        "bio_qa": 1,
        "rna_annotation": 2,
        "rna_function_prediction": 3,
        "protein_function_classification": 4,
        "text_classification": 5,
        "instruction_tuning": 6,
    },
    "sequence_type": {"BioText": 1, "RNA": 2, "DNA": 3, "Protein": 4, "Text": 5},
    "evaluation_metric": {"EM": 1, "F1-score": 2, "Accuracy": 3, "AUC": 4, "MAE": 5, "ROUGE": 6},
}

@dataclass
class RequirementVector:
    learning_rate: float
    epochs: int
    lr_scheduler: str
    finetune_method: str
    max_seq_length: int
    padding_strategy: str
    task_type: str
    sequence_type: str
    evaluation_metric: str


def load_requirement_text(requirement: Optional[str] = None, requirement_path: Optional[str] = None) -> str:
    if requirement and requirement.strip():
        return requirement.strip()

    if not requirement_path:
        raise ValueError("Either requirement or requirement_path must be provided.")

    with open(requirement_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        raise ValueError(f"Requirement file is empty: {requirement_path}")

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return raw_text

    if isinstance(payload, str):
        return payload.strip()

    if isinstance(payload, dict):
        for key in ("requirement", "requirements", "text", "prompt", "input", "instruction"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(payload, ensure_ascii=False)

    if isinstance(payload, list):
        parts = []
        for item in payload:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                for key in ("requirement", "text", "prompt", "input", "instruction"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
                        break
        if parts:
            return "\n".join(parts)

    return json.dumps(payload, ensure_ascii=False)


def load_knowledge_graph_context(knowledge_graph_path: Optional[str] = None) -> Tuple[str, Optional[str], Optional[str]]:
    path = knowledge_graph_path or DEFAULT_KNOWLEDGE_GRAPH_PATH
    if not path:
        return "", None, "Knowledge graph path is empty."

    if not os.path.exists(path):
        return "", path, f"Knowledge graph file is not accessible: {path}"

    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        return "", path, f"Knowledge graph file is empty: {path}"

    if len(raw_text) > MAX_KNOWLEDGE_GRAPH_CHARS:
        raw_text = raw_text[:MAX_KNOWLEDGE_GRAPH_CHARS]
        warning = (
            f"Knowledge graph context was truncated to {MAX_KNOWLEDGE_GRAPH_CHARS} characters "
            f"for prompt safety: {path}"
        )
    else:
        warning = None

    return raw_text, path, warning


def _contains_any(text: str, patterns: List[str]) -> bool:
    return any(pattern.lower() in text.lower() for pattern in patterns)


def _extract_gpu_memory_gb(text: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*(?:g|gb|G|GB)\s*(?:显存|显卡|GPU|gpu)?", text)
    if not match:
        return None
    return int(match.group(1))


def infer_requirement_vector(requirement_text: str) -> Tuple[RequirementVector, List[str]]:
    text = requirement_text or ""
    reasons = []

    low_data = _contains_any(text, ["样本少", "小数据", "数据规模偏小", "少量数据", "数据稀缺", "low data", "few-shot", "few shot", "small dataset"])
    effect_first = _contains_any(text, ["效果优先", "精度优先", "性能最优", "高准确率", "高召回", "最终部署", "effect first", "quality first", "accuracy first"])
    fast_first = _contains_any(text, ["快速", "缩短训练", "调试", "试跑", "验证流程", "训练快", "fast", "quick", "debug"])
    avoid_overfit = _contains_any(text, ["过拟合", "泛化", "稳定", "正则", "early stopping", "dropout", "overfit", "generalization", "stable"])
    gpu_memory = _extract_gpu_memory_gb(text)
    low_vram = bool(gpu_memory and gpu_memory <= 16) or _contains_any(text, ["显存受限", "低显存", "显存不足", "单卡", "消费级显卡"])

    if _contains_any(text, ["RNA", "rna", "非编码RNA", "ncRNA", "转录组"]):
        sequence_type = "RNA"
        task_type = "rna_function_prediction" if _contains_any(text, ["功能预测", "预测"]) else "rna_annotation"
        evaluation_metric = "AUC" if _contains_any(text, ["预测", "二分类", "分类"]) else "F1-score"
        max_seq_length = 512
        reasons.append("识别到 RNA/ncRNA 相关需求，选择 RNA 序列任务参数。")
    elif _contains_any(text, ["DNA", "基因组", "启动子", "motif"]):
        sequence_type = "DNA"
        task_type = "text_classification"
        evaluation_metric = "F1-score"
        max_seq_length = 1024
        reasons.append("识别到 DNA/基因组相关需求，选择序列分类参数。")
    elif _contains_any(text, ["蛋白", "protein", "氨基酸"]):
        sequence_type = "Protein"
        task_type = "protein_function_classification"
        evaluation_metric = "F1-score"
        max_seq_length = 1024
        reasons.append("识别到蛋白质相关需求，选择蛋白功能分类参数。")
    elif _contains_any(text, ["问答", "QA", "知识库", "医学问答"]):
        sequence_type = "BioText"
        task_type = "bio_qa"
        evaluation_metric = "EM"
        max_seq_length = 1024
        reasons.append("识别到问答类需求，选择生物医学文本问答参数。")
    else:
        sequence_type = "Text"
        task_type = "instruction_tuning"
        evaluation_metric = "F1-score"
        max_seq_length = 1024
        reasons.append("未识别到明确生物序列类型，按通用指令微调处理。")

    if low_vram:
        finetune_method = "QLoRA"
        max_seq_length = min(max_seq_length, 512)
        reasons.append("识别到显存受限，选择 QLoRA 并控制序列长度。")
    else:
        finetune_method = "LoRA"
        reasons.append("未发现强显存约束，选择 LoRA 作为默认高效微调方法。")

    if effect_first and not low_vram:
        learning_rate = 0.00003
        epochs = 10
        lr_scheduler = "cosine"
        reasons.append("需求强调效果优先，降低学习率并增加训练轮数。")
    elif effect_first:
        learning_rate = 0.00004
        epochs = 8
        lr_scheduler = "cosine"
        reasons.append("需求强调效果优先但资源受限，采用较低学习率和中等轮数。")
    elif fast_first:
        learning_rate = 0.00015
        epochs = 3
        lr_scheduler = "linear"
        reasons.append("需求强调快速验证，采用较高学习率和较少轮数。")
    elif low_data or avoid_overfit:
        learning_rate = 0.00005
        epochs = 5 if low_data else 7
        lr_scheduler = "cosine" if avoid_overfit else "constant"
        reasons.append("识别到小数据或泛化稳定性约束，采用保守学习率。")
    else:
        learning_rate = 0.00008
        epochs = 6
        lr_scheduler = "cosine_with_restarts"
        reasons.append("使用通用稳定配置作为默认微调参数。")

    return RequirementVector(
        learning_rate=round(learning_rate, 6),
        epochs=epochs,
        lr_scheduler=lr_scheduler,
        finetune_method=finetune_method,
        max_seq_length=max_seq_length,
        padding_strategy="max_length",
        task_type=task_type,
        sequence_type=sequence_type,
        evaluation_metric=evaluation_metric,
    ), reasons


def normalize_vector_payload(payload: Dict[str, Any]) -> RequirementVector:
    aliases = {
        "学习率": "learning_rate",
        "训练轮数": "epochs",
        "学习率衰减": "lr_scheduler",
        "微调方法": "finetune_method",
        "最大序列长度": "max_seq_length",
        "填充策略": "padding_strategy",
        "任务类型": "task_type",
        "序列类型": "sequence_type",
        "最优模型评价指标": "evaluation_metric",
    }
    normalized = {}
    for key, value in payload.items():
        normalized[aliases.get(key, key)] = value

    return RequirementVector(
        learning_rate=round(float(normalized["learning_rate"]), 6),
        epochs=int(normalized["epochs"]),
        lr_scheduler=str(normalized["lr_scheduler"]).strip(),
        finetune_method=str(normalized["finetune_method"]).strip(),
        max_seq_length=int(normalized["max_seq_length"]),
        padding_strategy=str(normalized["padding_strategy"]).strip(),
        task_type=str(normalized["task_type"]).strip(),
        sequence_type=str(normalized["sequence_type"]).strip(),
        evaluation_metric=str(normalized["evaluation_metric"]).strip(),
    )


def encode_vector(vector: RequirementVector) -> List[Any]:
    payload = asdict(vector)
    return [
        payload["learning_rate"],
        payload["epochs"],
        ENCODERS["lr_scheduler"].get(payload["lr_scheduler"], 0),
        ENCODERS["finetune_method"].get(payload["finetune_method"], 0),
        payload["max_seq_length"],
        ENCODERS["padding_strategy"].get(payload["padding_strategy"], 0),
        ENCODERS["task_type"].get(payload["task_type"], 0),
        ENCODERS["sequence_type"].get(payload["sequence_type"], 0),
        ENCODERS["evaluation_metric"].get(payload["evaluation_metric"], 0),
    ]


def vector_to_zh(vector: RequirementVector) -> Dict[str, Any]:
    payload = asdict(vector)
    return {LABELS_ZH[key]: payload[key] for key in VECTOR_FIELDS}


def _extract_json_object(text: str) -> Dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Model output does not contain a JSON object.")
    return json.loads(match.group(0))


def _build_prompt(
    requirement_text: str,
    fallback_vector: RequirementVector,
    knowledge_graph_context: str = "",
    previous_training_info: str = "",  # 新增的上一轮训练信息字段
) -> str:
    schema = {
        "learning_rate": "float, 1e-6 to 5e-4",
        "epochs": "int, 1 to 20",
        "lr_scheduler": list(ENCODERS["lr_scheduler"].keys()),
        "finetune_method": list(ENCODERS["finetune_method"].keys()),
        "max_seq_length": [256, 512, 1024, 2048, 4096],
        "padding_strategy": list(ENCODERS["padding_strategy"].keys()),
        "task_type": list(ENCODERS["task_type"].keys()),
        "sequence_type": list(ENCODERS["sequence_type"].keys()),
        "evaluation_metric": list(ENCODERS["evaluation_metric"].keys()),
    }
    knowledge_section = (
        "Knowledge graph from tupu.txt:\n"
        f"{knowledge_graph_context}\n\n"
        "How to use the knowledge graph:\n"
        "1. Read entities, related features, default values, and matched parameters from the graph.\n"
        "2. Prefer graph-supported parameter values when they do not conflict with the user's requirement.\n"
        "3. If multiple graph entries match, choose the parameter set with the strongest semantic overlap.\n"
        "4. If the graph is noisy or incomplete, combine it with the rule-based reference vector.\n\n"
        if knowledge_graph_context
        else "Knowledge graph from tupu.txt: not available. Use the rule-based reference vector.\n\n"
    )

    # 判断是否有 flag 中的上轮记录，如果有则替换系统提示词为优化指令
    if previous_training_info:
        system_instruction = (
        "You are a fine-tuning hyperparameter optimization agent. "
        "The previous fine-tuning round failed to satisfy the user's requirements. "
        "Please optimize the hyperparameters for the next training round based on the previous fine-tuning results.\n\n"
        f"The following information is from the previous round (including hyperparameter settings, evaluation accuracy, and training logs):\n"
        f"{previous_training_info}\n\n"
        "Please combine the historical information above with the current user requirement "
        "to generate a new hyperparameter vector in exactly the same JSON format as before.\n"
        "Return exactly one JSON object only. Do not output explanations or any extra text.\n"
    )
    else:
        system_instruction = (
            "You are a fine-tuning requirement analysis agent. Convert the user's natural-language "
            "fine-tuning requirement into exactly one JSON object. Do not output explanations. "
            "You must parse the knowledge graph context first, then use it to assist the final vector decision.\n\n"
        )

    return (
        f"{system_instruction}"
        f"Allowed schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"{knowledge_section}"
        f"Rule-based reference vector:\n{json.dumps(asdict(fallback_vector), ensure_ascii=False)}\n\n"
        f"User requirement:\n{requirement_text}\n"
    )


def parse_with_vllm(
    requirement_text: str,
    fallback_vector: RequirementVector,
    knowledge_graph_context: str = "",
    vllm_base_url: str = DEFAULT_VLLM_BASE_URL,
    vllm_model: str = DEFAULT_VLLM_MODEL,
    vllm_api_key: str = DEFAULT_VLLM_API_KEY,
    timeout: int = 120,
    previous_training_info: str = "",  # 新增传递上轮参数的字段
) -> RequirementVector:
    prompt = _build_prompt(
        requirement_text, 
        fallback_vector, 
        knowledge_graph_context=knowledge_graph_context,
        previous_training_info=previous_training_info
    )

    base_url = (vllm_base_url or DEFAULT_VLLM_BASE_URL).rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": vllm_model or DEFAULT_VLLM_MODEL,
        "messages": [
            {"role": "system", "content": "Return one valid JSON object only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 256,
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if vllm_api_key:
        headers["Authorization"] = f"Bearer {vllm_api_key}"

    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot connect to vLLM endpoint {url}: {exc.reason}") from exc

    try:
        raw = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected vLLM response format: {response_payload}") from exc

    return normalize_vector_payload(_extract_json_object(raw))


def analyze_requirement(
    requirement: Optional[str] = None,
    requirement_path: Optional[str] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    use_model: bool = False,
    knowledge_graph_path: Optional[str] = None,
    vllm_base_url: str = DEFAULT_VLLM_BASE_URL,
    vllm_model: str = DEFAULT_VLLM_MODEL,
    vllm_api_key: str = DEFAULT_VLLM_API_KEY,
) -> Dict[str, Any]:
    requirement_text = load_requirement_text(requirement=requirement, requirement_path=requirement_path)
    fallback_vector, reasons = infer_requirement_vector(requirement_text)
    knowledge_graph_context, resolved_knowledge_graph_path, kg_warning = load_knowledge_graph_context(knowledge_graph_path)
    used_model = False
    warnings = []
    
    if kg_warning:
        warnings.append(kg_warning)

    # ==== 新增逻辑：检测是否存在 flag.txt 以及读取上轮信息 ====
    has_flag = os.path.exists(DEFAULT_FLAG_PATH)

    if has_flag:
    # 先删除 flag.txt，避免重复触发下一轮
        try:
            os.remove(DEFAULT_FLAG_PATH)
        except Exception as e:
            warnings.append(f"Failed to remove {DEFAULT_FLAG_PATH}: {e}")

    previous_training_info = ""
    if has_flag and os.path.exists(DEFAULT_TRAINING_LAST_PATH):
        try:
            with open(DEFAULT_TRAINING_LAST_PATH, "r", encoding="utf-8") as f:
                previous_training_info = f.read().strip()
        except Exception as e:
            warnings.append(f"Failed to read {DEFAULT_TRAINING_LAST_PATH}: {e}")

    vector = fallback_vector
    if use_model:
        try:
            vector = parse_with_vllm(
                requirement_text,
                fallback_vector,
                knowledge_graph_context=knowledge_graph_context,
                vllm_base_url=vllm_base_url,
                vllm_model=vllm_model,
                vllm_api_key=vllm_api_key,
                previous_training_info=previous_training_info, # 将上轮信息传入请求
            )
            used_model = True
        except Exception as exc:
            warnings.append(f"vLLM parsing skipped: {exc}")
            vector = fallback_vector

    # ==== 新增逻辑：生成这轮超参数后，若有flag则清空文件并写入新参数 ====
    if has_flag:
        try:
            with open(DEFAULT_TRAINING_LAST_PATH, "w", encoding="utf-8") as f:
                # 按照约定格式将新的超参数向量存入
                f.write(f"Last Round Fine-Tuning Parameters: {json.dumps(asdict(vector), ensure_ascii=False)}")
        except Exception as e:
            warnings.append(f"Failed to write into {DEFAULT_TRAINING_LAST_PATH}: {e}")

    return {
        "requirement": requirement_text,
        "model_path": model_path,
        "vllm_base_url": vllm_base_url,
        "vllm_model": vllm_model,
        "knowledge_graph_path": resolved_knowledge_graph_path,
        "used_knowledge_graph": bool(knowledge_graph_context),
        "used_model": used_model,
        "fields": VECTOR_FIELDS,
        "labels_zh": LABELS_ZH,
        "params": asdict(vector),
        "params_zh": vector_to_zh(vector),
        "vector": encode_vector(vector),
        "encoders": ENCODERS,
        "reasons": reasons,
        "warnings": warnings,
    }


# import json
# import os
# import re
# import urllib.error
# import urllib.request
# from dataclasses import asdict, dataclass
# from typing import Any, Dict, List, Optional, Tuple


# DEFAULT_MODEL_PATH = "/private/HIT-ZLY/Qwen"
# DEFAULT_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
# DEFAULT_VLLM_MODEL = os.environ.get("VLLM_MODEL", "")
# DEFAULT_VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")
# DEFAULT_KNOWLEDGE_GRAPH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tupu.txt")
# MAX_KNOWLEDGE_GRAPH_CHARS = 12000

# VECTOR_FIELDS = [
#     "learning_rate",
#     "epochs",
#     "lr_scheduler",
#     "finetune_method",
#     "max_seq_length",
#     "padding_strategy",
#     "task_type",
#     "sequence_type",
#     "evaluation_metric",
# ]

# LABELS_ZH = {
#     "learning_rate": "学习率",
#     "epochs": "训练轮数",
#     "lr_scheduler": "学习率衰减",
#     "finetune_method": "微调方法",
#     "max_seq_length": "最大序列长度",
#     "padding_strategy": "填充策略",
#     "task_type": "任务类型",
#     "sequence_type": "序列类型",
#     "evaluation_metric": "最优模型评价指标",
# }

# ENCODERS = {
#     "lr_scheduler": {"constant": 1, "linear": 2, "cosine": 3, "cosine_with_restarts": 4},
#     "finetune_method": {"LoRA": 1, "QLoRA": 2, "Full fine-tuning": 3},
#     "padding_strategy": {"max_length": 1, "longest": 2, "dynamic": 3},
#     "task_type": {
#         "bio_qa": 1,
#         "rna_annotation": 2,
#         "rna_function_prediction": 3,
#         "protein_function_classification": 4,
#         "text_classification": 5,
#         "instruction_tuning": 6,
#     },
#     "sequence_type": {"BioText": 1, "RNA": 2, "DNA": 3, "Protein": 4, "Text": 5},
#     "evaluation_metric": {"EM": 1, "F1-score": 2, "Accuracy": 3, "AUC": 4, "MAE": 5, "ROUGE": 6},
# }


# @dataclass
# class RequirementVector:
#     learning_rate: float
#     epochs: int
#     lr_scheduler: str
#     finetune_method: str
#     max_seq_length: int
#     padding_strategy: str
#     task_type: str
#     sequence_type: str
#     evaluation_metric: str


# def load_requirement_text(requirement: Optional[str] = None, requirement_path: Optional[str] = None) -> str:
#     if requirement and requirement.strip():
#         return requirement.strip()

#     if not requirement_path:
#         raise ValueError("Either requirement or requirement_path must be provided.")

#     with open(requirement_path, "r", encoding="utf-8") as f:
#         raw_text = f.read().strip()

#     if not raw_text:
#         raise ValueError(f"Requirement file is empty: {requirement_path}")

#     try:
#         payload = json.loads(raw_text)
#     except json.JSONDecodeError:
#         return raw_text

#     if isinstance(payload, str):
#         return payload.strip()

#     if isinstance(payload, dict):
#         for key in ("requirement", "requirements", "text", "prompt", "input", "instruction"):
#             value = payload.get(key)
#             if isinstance(value, str) and value.strip():
#                 return value.strip()
#         return json.dumps(payload, ensure_ascii=False)

#     if isinstance(payload, list):
#         parts = []
#         for item in payload:
#             if isinstance(item, str) and item.strip():
#                 parts.append(item.strip())
#             elif isinstance(item, dict):
#                 for key in ("requirement", "text", "prompt", "input", "instruction"):
#                     value = item.get(key)
#                     if isinstance(value, str) and value.strip():
#                         parts.append(value.strip())
#                         break
#         if parts:
#             return "\n".join(parts)

#     return json.dumps(payload, ensure_ascii=False)


# def load_knowledge_graph_context(knowledge_graph_path: Optional[str] = None) -> Tuple[str, Optional[str], Optional[str]]:
#     path = knowledge_graph_path or DEFAULT_KNOWLEDGE_GRAPH_PATH
#     if not path:
#         return "", None, "Knowledge graph path is empty."

#     if not os.path.exists(path):
#         return "", path, f"Knowledge graph file is not accessible: {path}"

#     with open(path, "r", encoding="utf-8") as f:
#         raw_text = f.read().strip()

#     if not raw_text:
#         return "", path, f"Knowledge graph file is empty: {path}"

#     if len(raw_text) > MAX_KNOWLEDGE_GRAPH_CHARS:
#         raw_text = raw_text[:MAX_KNOWLEDGE_GRAPH_CHARS]
#         warning = (
#             f"Knowledge graph context was truncated to {MAX_KNOWLEDGE_GRAPH_CHARS} characters "
#             f"for prompt safety: {path}"
#         )
#     else:
#         warning = None

#     return raw_text, path, warning


# def _contains_any(text: str, patterns: List[str]) -> bool:
#     return any(pattern.lower() in text.lower() for pattern in patterns)


# def _extract_gpu_memory_gb(text: str) -> Optional[int]:
#     match = re.search(r"(\d+)\s*(?:g|gb|G|GB)\s*(?:显存|显卡|GPU|gpu)?", text)
#     if not match:
#         return None
#     return int(match.group(1))


# def infer_requirement_vector(requirement_text: str) -> Tuple[RequirementVector, List[str]]:
#     text = requirement_text or ""
#     reasons = []

#     low_data = _contains_any(text, ["样本少", "小数据", "数据规模偏小", "少量数据", "数据稀缺", "low data", "few-shot", "few shot", "small dataset"])
#     effect_first = _contains_any(text, ["效果优先", "精度优先", "性能最优", "高准确率", "高召回", "最终部署", "effect first", "quality first", "accuracy first"])
#     fast_first = _contains_any(text, ["快速", "缩短训练", "调试", "试跑", "验证流程", "训练快", "fast", "quick", "debug"])
#     avoid_overfit = _contains_any(text, ["过拟合", "泛化", "稳定", "正则", "early stopping", "dropout", "overfit", "generalization", "stable"])
#     gpu_memory = _extract_gpu_memory_gb(text)
#     low_vram = bool(gpu_memory and gpu_memory <= 16) or _contains_any(text, ["显存受限", "低显存", "显存不足", "单卡", "消费级显卡"])

#     if _contains_any(text, ["RNA", "rna", "非编码RNA", "ncRNA", "转录组"]):
#         sequence_type = "RNA"
#         task_type = "rna_function_prediction" if _contains_any(text, ["功能预测", "预测"]) else "rna_annotation"
#         evaluation_metric = "AUC" if _contains_any(text, ["预测", "二分类", "分类"]) else "F1-score"
#         max_seq_length = 512
#         reasons.append("识别到 RNA/ncRNA 相关需求，选择 RNA 序列任务参数。")
#     elif _contains_any(text, ["DNA", "基因组", "启动子", "motif"]):
#         sequence_type = "DNA"
#         task_type = "text_classification"
#         evaluation_metric = "F1-score"
#         max_seq_length = 1024
#         reasons.append("识别到 DNA/基因组相关需求，选择序列分类参数。")
#     elif _contains_any(text, ["蛋白", "protein", "氨基酸"]):
#         sequence_type = "Protein"
#         task_type = "protein_function_classification"
#         evaluation_metric = "F1-score"
#         max_seq_length = 1024
#         reasons.append("识别到蛋白质相关需求，选择蛋白功能分类参数。")
#     elif _contains_any(text, ["问答", "QA", "知识库", "医学问答"]):
#         sequence_type = "BioText"
#         task_type = "bio_qa"
#         evaluation_metric = "EM"
#         max_seq_length = 1024
#         reasons.append("识别到问答类需求，选择生物医学文本问答参数。")
#     else:
#         sequence_type = "Text"
#         task_type = "instruction_tuning"
#         evaluation_metric = "F1-score"
#         max_seq_length = 1024
#         reasons.append("未识别到明确生物序列类型，按通用指令微调处理。")

#     if low_vram:
#         finetune_method = "QLoRA"
#         max_seq_length = min(max_seq_length, 512)
#         reasons.append("识别到显存受限，选择 QLoRA 并控制序列长度。")
#     else:
#         finetune_method = "LoRA"
#         reasons.append("未发现强显存约束，选择 LoRA 作为默认高效微调方法。")

#     if effect_first and not low_vram:
#         learning_rate = 0.00003
#         epochs = 10
#         lr_scheduler = "cosine"
#         reasons.append("需求强调效果优先，降低学习率并增加训练轮数。")
#     elif effect_first:
#         learning_rate = 0.00004
#         epochs = 8
#         lr_scheduler = "cosine"
#         reasons.append("需求强调效果优先但资源受限，采用较低学习率和中等轮数。")
#     elif fast_first:
#         learning_rate = 0.00015
#         epochs = 3
#         lr_scheduler = "linear"
#         reasons.append("需求强调快速验证，采用较高学习率和较少轮数。")
#     elif low_data or avoid_overfit:
#         learning_rate = 0.00005
#         epochs = 5 if low_data else 7
#         lr_scheduler = "cosine" if avoid_overfit else "constant"
#         reasons.append("识别到小数据或泛化稳定性约束，采用保守学习率。")
#     else:
#         learning_rate = 0.00008
#         epochs = 6
#         lr_scheduler = "cosine_with_restarts"
#         reasons.append("使用通用稳定配置作为默认微调参数。")

#     return RequirementVector(
#         learning_rate=round(learning_rate, 6),
#         epochs=epochs,
#         lr_scheduler=lr_scheduler,
#         finetune_method=finetune_method,
#         max_seq_length=max_seq_length,
#         padding_strategy="max_length",
#         task_type=task_type,
#         sequence_type=sequence_type,
#         evaluation_metric=evaluation_metric,
#     ), reasons


# def normalize_vector_payload(payload: Dict[str, Any]) -> RequirementVector:
#     aliases = {
#         "学习率": "learning_rate",
#         "训练轮数": "epochs",
#         "学习率衰减": "lr_scheduler",
#         "微调方法": "finetune_method",
#         "最大序列长度": "max_seq_length",
#         "填充策略": "padding_strategy",
#         "任务类型": "task_type",
#         "序列类型": "sequence_type",
#         "最优模型评价指标": "evaluation_metric",
#     }
#     normalized = {}
#     for key, value in payload.items():
#         normalized[aliases.get(key, key)] = value

#     return RequirementVector(
#         learning_rate=round(float(normalized["learning_rate"]), 6),
#         epochs=int(normalized["epochs"]),
#         lr_scheduler=str(normalized["lr_scheduler"]).strip(),
#         finetune_method=str(normalized["finetune_method"]).strip(),
#         max_seq_length=int(normalized["max_seq_length"]),
#         padding_strategy=str(normalized["padding_strategy"]).strip(),
#         task_type=str(normalized["task_type"]).strip(),
#         sequence_type=str(normalized["sequence_type"]).strip(),
#         evaluation_metric=str(normalized["evaluation_metric"]).strip(),
#     )


# def encode_vector(vector: RequirementVector) -> List[Any]:
#     payload = asdict(vector)
#     return [
#         payload["learning_rate"],
#         payload["epochs"],
#         ENCODERS["lr_scheduler"].get(payload["lr_scheduler"], 0),
#         ENCODERS["finetune_method"].get(payload["finetune_method"], 0),
#         payload["max_seq_length"],
#         ENCODERS["padding_strategy"].get(payload["padding_strategy"], 0),
#         ENCODERS["task_type"].get(payload["task_type"], 0),
#         ENCODERS["sequence_type"].get(payload["sequence_type"], 0),
#         ENCODERS["evaluation_metric"].get(payload["evaluation_metric"], 0),
#     ]


# def vector_to_zh(vector: RequirementVector) -> Dict[str, Any]:
#     payload = asdict(vector)
#     return {LABELS_ZH[key]: payload[key] for key in VECTOR_FIELDS}


# def _extract_json_object(text: str) -> Dict[str, Any]:
#     match = re.search(r"\{[\s\S]*\}", text)
#     if not match:
#         raise ValueError("Model output does not contain a JSON object.")
#     return json.loads(match.group(0))


# def _build_prompt(
#     requirement_text: str,
#     fallback_vector: RequirementVector,
#     knowledge_graph_context: str = "",
# ) -> str:
#     schema = {
#         "learning_rate": "float, 1e-6 to 5e-4",
#         "epochs": "int, 1 to 20",
#         "lr_scheduler": list(ENCODERS["lr_scheduler"].keys()),
#         "finetune_method": list(ENCODERS["finetune_method"].keys()),
#         "max_seq_length": [256, 512, 1024, 2048, 4096],
#         "padding_strategy": list(ENCODERS["padding_strategy"].keys()),
#         "task_type": list(ENCODERS["task_type"].keys()),
#         "sequence_type": list(ENCODERS["sequence_type"].keys()),
#         "evaluation_metric": list(ENCODERS["evaluation_metric"].keys()),
#     }
#     knowledge_section = (
#         "Knowledge graph from tupu.txt:\n"
#         f"{knowledge_graph_context}\n\n"
#         "How to use the knowledge graph:\n"
#         "1. Read entities, related features, default values, and matched parameters from the graph.\n"
#         "2. Prefer graph-supported parameter values when they do not conflict with the user's requirement.\n"
#         "3. If multiple graph entries match, choose the parameter set with the strongest semantic overlap.\n"
#         "4. If the graph is noisy or incomplete, combine it with the rule-based reference vector.\n\n"
#         if knowledge_graph_context
#         else "Knowledge graph from tupu.txt: not available. Use the rule-based reference vector.\n\n"
#     )

#     return (
#         "You are a fine-tuning requirement analysis agent. Convert the user's natural-language "
#         "fine-tuning requirement into exactly one JSON object. Do not output explanations. "
#         "You must parse the knowledge graph context first, then use it to assist the final vector decision.\n\n"
#         f"Allowed schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
#         f"{knowledge_section}"
#         f"Rule-based reference vector:\n{json.dumps(asdict(fallback_vector), ensure_ascii=False)}\n\n"
#         f"User requirement:\n{requirement_text}\n"
#     )


# def parse_with_vllm(
#     requirement_text: str,
#     fallback_vector: RequirementVector,
#     knowledge_graph_context: str = "",
#     vllm_base_url: str = DEFAULT_VLLM_BASE_URL,
#     vllm_model: str = DEFAULT_VLLM_MODEL,
#     vllm_api_key: str = DEFAULT_VLLM_API_KEY,
#     timeout: int = 120,
# ) -> RequirementVector:
#     prompt = _build_prompt(requirement_text, fallback_vector, knowledge_graph_context=knowledge_graph_context)

#     base_url = (vllm_base_url or DEFAULT_VLLM_BASE_URL).rstrip("/")
#     url = f"{base_url}/chat/completions"
#     payload = {
#         "model": vllm_model or DEFAULT_VLLM_MODEL,
#         "messages": [
#             {"role": "system", "content": "Return one valid JSON object only."},
#             {"role": "user", "content": prompt},
#         ],
#         "temperature": 0,
#         "max_tokens": 256,
#     }

#     data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
#     headers = {"Content-Type": "application/json"}
#     if vllm_api_key:
#         headers["Authorization"] = f"Bearer {vllm_api_key}"

#     request = urllib.request.Request(url, data=data, headers=headers, method="POST")
#     try:
#         with urllib.request.urlopen(request, timeout=timeout) as response:
#             response_payload = json.loads(response.read().decode("utf-8"))
#     except urllib.error.HTTPError as exc:
#         error_body = exc.read().decode("utf-8", errors="replace")
#         raise RuntimeError(f"vLLM HTTP {exc.code}: {error_body}") from exc
#     except urllib.error.URLError as exc:
#         raise RuntimeError(f"Cannot connect to vLLM endpoint {url}: {exc.reason}") from exc

#     try:
#         raw = response_payload["choices"][0]["message"]["content"]
#     except (KeyError, IndexError, TypeError) as exc:
#         raise RuntimeError(f"Unexpected vLLM response format: {response_payload}") from exc

#     return normalize_vector_payload(_extract_json_object(raw))


# def analyze_requirement(
#     requirement: Optional[str] = None,
#     requirement_path: Optional[str] = None,
#     model_path: str = DEFAULT_MODEL_PATH,
#     use_model: bool = False,
#     knowledge_graph_path: Optional[str] = None,
#     vllm_base_url: str = DEFAULT_VLLM_BASE_URL,
#     vllm_model: str = DEFAULT_VLLM_MODEL,
#     vllm_api_key: str = DEFAULT_VLLM_API_KEY,
# ) -> Dict[str, Any]:
#     requirement_text = load_requirement_text(requirement=requirement, requirement_path=requirement_path)
#     fallback_vector, reasons = infer_requirement_vector(requirement_text)
#     knowledge_graph_context, resolved_knowledge_graph_path, kg_warning = load_knowledge_graph_context(knowledge_graph_path)
#     used_model = False
#     warnings = []
#     if kg_warning:
#         warnings.append(kg_warning)

#     vector = fallback_vector
#     if use_model:
#         try:
#             vector = parse_with_vllm(
#                 requirement_text,
#                 fallback_vector,
#                 knowledge_graph_context=knowledge_graph_context,
#                 vllm_base_url=vllm_base_url,
#                 vllm_model=vllm_model,
#                 vllm_api_key=vllm_api_key,
#             )
#             used_model = True
#         except Exception as exc:
#             warnings.append(f"vLLM parsing skipped: {exc}")
#             vector = fallback_vector

#     return {
#         "requirement": requirement_text,
#         "model_path": model_path,
#         "vllm_base_url": vllm_base_url,
#         "vllm_model": vllm_model,
#         "knowledge_graph_path": resolved_knowledge_graph_path,
#         "used_knowledge_graph": bool(knowledge_graph_context),
#         "used_model": used_model,
#         "fields": VECTOR_FIELDS,
#         "labels_zh": LABELS_ZH,
#         "params": asdict(vector),
#         "params_zh": vector_to_zh(vector),
#         "vector": encode_vector(vector),
#         "encoders": ENCODERS,
#         "reasons": reasons,
#         "warnings": warnings,
#     }