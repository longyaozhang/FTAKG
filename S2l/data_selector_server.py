import logging
import os
import sys
import subprocess
import yaml
import tempfile
import asyncio
import torch
from typing import Any, Dict

from mcp.server import FastMCP
from mcp.types import TextContent

# ==========================================
# [配置 1] 解决 Windows 控制台乱码
# ==========================================
if sys.platform == 'win32':
    try:
        # 强制 stderr 使用系统默认编码 (如 gbk/mbcs)，防止 UTF-8 字符在 GBK 终端乱码
        sys.stderr.reconfigure(encoding='mbcs', errors='replace')
    except Exception:
        pass

# [配置 2] 重置日志 Handler，确保日志只输出到 stderr (MCP 协议要求)
root_logger = logging.getLogger()
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

log_handler = logging.StreamHandler(sys.stderr)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root_logger.addHandler(log_handler)
root_logger.setLevel(logging.INFO)

logger = logging.getLogger("data_selector_server")
# ==========================================


mcp_server = FastMCP(
    name="data_selector_server",
    instructions="自动化数据集筛选服务",
)


async def run_command(cmd: list[str], description: str):
    """
    异步执行 Shell 命令，并实时流式输出日志
    """
    logger.info(f"开始执行: {description}")
    logger.info(f"CMD: {' '.join(cmd)}")

    def _streaming_exec():
        # 准备环境变量
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"  # 子进程内部使用 UTF-8
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决 OMP 报错
        env["PYTHONUNBUFFERED"] = "1"  # [关键] 强制禁用缓冲，实现实时输出

        # 使用 Popen 而不是 run，以便逐行读取输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 将错误输出合并到标准输出一起打印
            text=True,
            cwd=os.getcwd(),
            encoding='utf-8',
            errors='replace',
            env=env,
            bufsize=1  # 行缓冲
        )

        # 实时逐行读取并打印
        captured_output = []
        for line in process.stdout:
            line = line.strip()
            if line:
                # 打印到 MCP 服务器日志 (会显示在你的 s2l_agent 控制台)
                logger.info(f"[{description}] {line}")
                captured_output.append(line)

        # 等待进程结束
        process.wait()

        if process.returncode != 0:
            error_msg = f"{description} 失败，返回码: {process.returncode}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        return "\n".join(captured_output)

    try:
        # 放到线程池运行，不阻塞心跳
        await asyncio.to_thread(_streaming_exec)
        logger.info(f"{description} 执行成功。")
        return f"{description} 完成"
    except Exception as e:
        logger.error(f"执行异常: {str(e)}")
        raise e


def create_temp_config(base_config: Dict[str, Any]) -> str:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False, encoding='utf-8') as tmp:
        yaml.dump(base_config, tmp, sort_keys=False, default_flow_style=False)
        return tmp.name


@mcp_server.tool()
async def perform_data_selection(
        small_model_path: str,
        dataset_path: str,
        result_dir_name: str
) -> list[TextContent]:
    """
    执行全自动数据筛选流程：训练 -> 收集 -> 筛选
    """
    logger.info(f"收到数据筛选任务: {result_dir_name}")
    base_cache_dir: str = "./cache"
    # 通用参数
    common_train_args = {
        "optim": "adamw_torch",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "save_strategy": "steps",
        "save_steps": 500,
        "save_total_limit": 12,
        "learning_rate": 2.0e-5,
        "weight_decay": 0.,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "logging_steps": 1,
        "bf16": True,
        "tf32": False,
        "group_by_length": True,
        "full_determinism": True,
        "seed": 42
    }

    # Step 1 Config
    step1_dir_name = f"{result_dir_name}_step1_proxy"
    step1_config_data = {
        "full_data_path": dataset_path,
        "model_name_or_path": small_model_path,
        "cache_dir": base_cache_dir,
        "model_max_length": 512,
        "schedule_name": "Full",
        "result_dir_name": step1_dir_name,
        "train_args": common_train_args
    }
    step1_config_path = create_temp_config(step1_config_data)
    step1_output_model_path = os.path.join("res", step1_dir_name, "output")

    # Step 3 Config
    step3_config_data = step1_config_data.copy()
    step3_config_data["schedule_name"] = "S2L"
    step3_config_data["result_dir_name"] = result_dir_name
    step3_config_data["ref_model_path"] = step1_output_model_path
    step3_config_data["n_components"] = 10
    step3_config_data["init_label_num"] = 1200 # 筛选剩多少条数据
    step3_config_data["n_round"] = 0
    step3_config_data["n_query"] = 1200
    step3_config_path = create_temp_config(step3_config_data)

    gpu_count = torch.cuda.device_count()
    try:
        # Step 1: Train (实时输出日志)
        await run_command(
            [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={gpu_count}", "train.py", "--config_file", step1_config_path],
            "Step 1: 训练代理小模型"
        )

        # Step 2: Collect Trajectories
        step1_output_model_path = os.path.join("res", step1_dir_name, "output")
        await run_command(
            [sys.executable, "run_distributed_trajectories.py", "--model_path", step1_output_model_path,
             "--config_file", step1_config_path, "--checkpoints", "all"],
            "Step 2: 收集模型训练轨迹"
        )

        # Step 3: S2L Selection
        await run_command(
            [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={gpu_count}", "s2l_data_selector.py", "--config_file", step3_config_path],
            "Step 3: S2L 数据筛选"
        )

        final_data_path = os.path.join("res", result_dir_name, "data")
        msg = f"流程执行完毕。筛选数据已保存至: {final_data_path}"
        logger.info(msg)
        return [TextContent(type="text", text=msg)]

    finally:
        # 清理临时文件
        if os.path.exists(step1_config_path): os.remove(step1_config_path)
        if os.path.exists(step3_config_path): os.remove(step3_config_path)


if __name__ == "__main__":
    mcp_server.run(transport='stdio')