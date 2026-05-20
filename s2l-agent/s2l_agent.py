import asyncio
import sys

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import ModelInfo, ModelFamily

# 1. 配置 Ollama 客户端
model_client = OllamaChatCompletionClient(
    model="XXX",
    host="XXX",
    model_info=ModelInfo(
        vision=False,
        function_calling=True,  # 必须为 True 才能使用工具
        json_output=False,
        family=ModelFamily.UNKNOWN
    )
)

# 2. 配置 MCP Server (指向新的 data_selector_server.py)
data_selector_server = StdioServerParams(
    command=sys.executable,  # 使用当前 python 环境
    args=["data_selector_server.py"],
    read_timeout_seconds=3600
)


async def run_agent(task: str):
    # 3. 获取工具
    tools = await mcp_server_tools(data_selector_server)

    # 4. 创建智能体，配置 system_message
    agent = AssistantAgent(
        name="data_selection_assistant",
        model_client=model_client,
        tools=tools,
        reflect_on_tool_use=True,
        system_message="""
        你是一个专业的数据集筛选助手。
        你拥有工具 'perform_data_selection'，可以自动化执行小模型训练、轨迹收集和数据筛选。

        收到用户指令后，请提取以下参数并调用工具：
        - small_model_path (小模型路径)
        - dataset_path (原始数据集路径)
        - result_dir_name (实验结果目录名)

        如果缺少任何参数，请询问用户。
        """
    )

    # 5. 运行并打印结果
    await Console(
        agent.run_stream(task=task),
        output_stats=True,
    )

    task_prompt = """
    请帮我筛选数据。使用的小模型在: XXX。原始数据集在: XXX。这次实验结果请保存为: XXX。
    """
if __name__ == '__main__':
    # 逻辑修改：优先从命令行参数读取，如果没有则提示输入
    task_prompt = ""

    if len(sys.argv) > 1:
        # 方式 1: python s2l_agent.py "请帮我筛选数据..."
        task_prompt = " ".join(sys.argv[1:])
    else:
        # 方式 2: 直接运行 python s2l_agent.py，进入交互输入
        # print("Usage: python s2l_agent.py <your_prompt>")
        # print("-" * 50)
        print("请输入您的任务指令 (例如: 请帮我筛选数据，小模型路径在...):")
        try:
            task_prompt = input(">>> ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

    if not task_prompt:
        print("[Error] 未输入指令，程序退出。")
        sys.exit(1)

    asyncio.run(run_agent(task=task_prompt))