import os
import asyncio
from typing import Annotated, Literal
from typing_extensions import TypedDict

from mcp import ClientSession
from mcp.client.sse import sse_client

from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

class State(TypedDict):
    messages: Annotated[list, add_messages]

# ✅ 限制 max_tokens，避免炸显存
llm = ChatOpenAI(
    model="XXX",
    temperature=0,
    max_tokens=512,
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

# =========================
# MCP 调用
# =========================
async def call_remote_mcp_tool(url: str, tool_name: str, arguments: dict) -> str:
    sse_url = f"{url}/sse" if not url.endswith("/sse") else url
    try:
        async with sse_client(sse_url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                return result.content[0].text
    except Exception as e:
        return f"[ERROR] {url}: {str(e)}"

# =========================
# Tools
# =========================
# @tool
# async def data_filter_tool(query: str) -> str:
#     """负责根据用户的初始需求，检索并筛选合适的数据集"""
#     return await call_remote_mcp_tool("http://localhost:8093", "data_filter", {"query": query})

@tool
async def request_parser_tool(data_summary: str) -> str:
    """负责将筛选后的数据和需求，解析为标准的微调配置参数"""
    return await call_remote_mcp_tool("http://localhost:8094", "request_parser", {"data_summary": data_summary})

@tool
async def model_finetune_tool(config: str) -> str:
    """负责启动模型微调任务"""
    return await call_remote_mcp_tool("http://localhost:8095", "fine_tune_model", {"config": config})

@tool
async def model_evaluate_tool(model_path: str) -> str:
    """负责评估微调后的模型性能"""
    return await call_remote_mcp_tool("http://localhost:8096", "benchmark_model", {"model_path": model_path})

# =========================
# 核心执行逻辑（修复版）
# =========================
async def run_agent(state: State, prompt: str, tool_func):
    # 1️⃣ 让 LLM 只负责“生成工具参数”
    ai_msg = await llm.ainvoke([
        HumanMessage(content=prompt)
    ])

    # 2️⃣ 强制调用工具（不让 LLM 决策）
    tool_result = await tool_func.ainvoke(ai_msg.content)

    tool_msg = ToolMessage(
        content=str(tool_result),
        name=tool_func.name,
        tool_call_id="forced_call"
    )

    return {"messages": [ai_msg, tool_msg]}

# =========================
# Agents
# =========================
async def agent_1_filter(state: State):
    print("🚀 数据筛选")
    return await run_agent(
        state,
        f"请提取用于数据筛选的query：\n{state['messages'][-1].content}",
        data_filter_tool
    )

async def agent_2_parser(state: State):
    print("🧩 参数解析")
    return await run_agent(
        state,
        f"根据以下数据生成微调配置：\n{state['messages'][-1].content}",
        request_parser_tool
    )

async def agent_3_finetune(state: State):
    print("🔥 模型微调")
    return await run_agent(
        state,
        f"生成微调配置参数：\n{state['messages'][-1].content}",
        model_finetune_tool
    )

async def agent_4_evaluate(state: State):
    print("⚖️ 模型评估")
    return await run_agent(
        state,
        f"提取模型路径用于评估：\n{state['messages'][-1].content}",
        model_evaluate_tool
    )

def check_evaluation(state: State) -> Literal["agent_3_finetune", "__end__"]:
    if "FAIL" in state["messages"][-1].content:
        return "agent_3_finetune"
    return END

# =========================
# Graph
# =========================
workflow = StateGraph(State)

workflow.add_node("agent_1_filter", agent_1_filter)
workflow.add_node("agent_2_parser", agent_2_parser)
workflow.add_node("agent_3_finetune", agent_3_finetune)
workflow.add_node("agent_4_evaluate", agent_4_evaluate)

workflow.add_edge(START, "agent_1_filter")
workflow.add_edge(START, "agent_2_parser")
workflow.add_edge("agent_2_parser", "agent_3_finetune")
workflow.add_edge("agent_3_finetune", "agent_4_evaluate")
workflow.add_conditional_edges("agent_4_evaluate", check_evaluation)

graph = workflow.compile(checkpointer=MemorySaver())

# =========================
# 运行
# =========================
async def main():
    config = {"configurable": {"thread_id": "run_01"}}
    print("\n👉 请输入本次微调任务的配置：")
    dataset_path = input("📂 数据集地址: ") or "XXX"
    base_model_path = input("🧠 基座模型地址: ") or "XXX"
    output_path = input("💾 模型保存地址：") or "XXX"
    task_desc = input("📝 任务描述/特殊要求: ") or "XXX"
    small_model_path = input("🔍 数据筛选小模型地址: ") or "XXX"

    prompt = f"""
    任务描述：{task_desc}
    数据集: {dataset_path}
    基座模型: {base_model_path}
    输出路径: {output_path}
    小模型路径: {small_model_path}
    """

    async for _ in graph.astream(
        {"messages": [HumanMessage(content=prompt)]},
        config,
        stream_mode="values",
    ):
        pass

    print("✅ 完成")

if __name__ == "__main__":
    asyncio.run(main())

# import os
# import asyncio
# from typing import Annotated, Literal
# from typing_extensions import TypedDict

# # 引入官方 MCP Client 库
# from mcp import ClientSession
# from mcp.client.sse import sse_client

# from langchain_core.messages import ToolMessage, HumanMessage
# from langchain_core.tools import tool
# from langchain_openai import ChatOpenAI
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langgraph.checkpoint.memory import MemorySaver

# os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

# class State(TypedDict):
#     messages: Annotated[list, add_messages]

# llm = ChatOpenAI(
#     model="Qwen", 
#     temperature=0,
#     base_url="http://localhost:8000/v1",
#     api_key="EMPTY",
# )

# # ==========================================
# # 🌐 核心改造：通用的远程 MCP 调用包装器
# # ==========================================
# async def call_remote_mcp_tool(url: str, tool_name: str, arguments: dict) -> str:
#     """连接到指定的 MCP 服务端端口并执行工具"""
#     # FastMCP 提供 SSE 端点，通常路径为 /sse
#     sse_url = f"{url}/sse" if not url.endswith("/sse") else url
    
#     try:
#         async with sse_client(sse_url) as streams:
#             async with ClientSession(streams[0], streams[1]) as session:
#                 await session.initialize()
#                 # 调用远程工具
#                 result = await session.call_tool(tool_name, arguments)
#                 return result.content[0].text
#     except Exception as e:
#         return f"调用远程服务 {url} 失败: {str(e)}"

# # ==========================================
# # 🤖 将远程调用包装为 LangChain Tool
# # ==========================================
# @tool
# async def data_filter_tool(query: str) -> str:
#     """负责根据用户的初始需求，检索并筛选合适的数据集"""
#     return await call_remote_mcp_tool("http://localhost:8093", "data_filter_mcp", {"query": query})

# @tool
# async def request_parser_tool(data_summary: str) -> str:
#     """负责将筛选后的数据和需求，解析为标准的微调配置参数"""
#     return await call_remote_mcp_tool("http://localhost:8094", "request_parser_mcp", {"data_summary": data_summary})

# @tool
# async def model_finetune_tool(config: str) -> str:
#     """负责启动模型微调任务"""
#     return await call_remote_mcp_tool("http://localhost:8095", "model_finetune_mcp", {"config": config})

# @tool
# async def model_evaluate_tool(model_path: str) -> str:
#     """负责评估微调后的模型性能"""
#     return await call_remote_mcp_tool("http://localhost:8096", "model_evaluate_mcp", {"model_path": model_path})

# # ==========================================
# # 🧠 异步 LangGraph 工作流定义
# # ==========================================
# async def run_agent_with_tool(state: State, specific_tool) -> dict:
#     llm_with_tool = llm.bind_tools([specific_tool])
    
#     # 异步调用模型
#     ai_msg = await llm_with_tool.ainvoke(state["messages"])
    
#     tool_call = ai_msg.tool_calls[0]
#     # 异步执行工具 (触发网络请求到 MCP 服务端)
#     tool_result = await specific_tool.ainvoke(tool_call)
    
#     tool_msg = ToolMessage(
#         content=str(tool_result), 
#         name=tool_call["name"], 
#         tool_call_id=tool_call["id"]
#     )
#     return {"messages": [ai_msg, tool_msg]}

# # --- 定义异步智能体节点 ---
# async def agent_1_filter(state: State):
#     print("🚀 [Agent 1] 请求 8093 端口进行数据筛选...")
#     return await run_agent_with_tool(state, data_filter_tool)

# async def agent_2_parser(state: State):
#     print("🧩 [Agent 2] 请求 8094 端口解析微调配置...")
#     return await run_agent_with_tool(state, request_parser_tool)

# async def agent_3_finetune(state: State):
#     print("🔥 [Agent 3] 请求 8095 端口微调模型...")
#     return await run_agent_with_tool(state, model_finetune_tool)

# async def agent_4_evaluate(state: State):
#     print("⚖️  [Agent 4] 请求 8096 端口评估模型...")
#     return await run_agent_with_tool(state, model_evaluate_tool)

# def check_evaluation(state: State) -> Literal["agent_3_finetune", "__end__"]:
#     last_message = state["messages"][-1].content
#     if "STATUS: FAIL" in last_message:
#         return "agent_3_finetune"
#     print("✅ 评估通过，任务结束。")
#     return END

# # --- 构建图 ---
# workflow = StateGraph(State)
# workflow.add_node("agent_1_filter", agent_1_filter)
# workflow.add_node("agent_2_parser", agent_2_parser)
# workflow.add_node("agent_3_finetune", agent_3_finetune)
# workflow.add_node("agent_4_evaluate", agent_4_evaluate)

# workflow.add_edge(START, "agent_1_filter")
# workflow.add_edge("agent_1_filter", "agent_2_parser")
# workflow.add_edge("agent_2_parser", "agent_3_finetune")
# workflow.add_edge("agent_3_finetune", "agent_4_evaluate")
# workflow.add_conditional_edges("agent_4_evaluate", check_evaluation)

# memory = MemorySaver()
# graph = workflow.compile(checkpointer=memory)
# # # ==========================================
# # # 📸 新增：可视化并输出 LangGraph 工作流图
# # # ==========================================
# # print("\n🎨 正在生成工作流结构图...")
# # try:
# #     # 尝试将图渲染为 PNG 图片并保存到本地
# #     image_data = graph.get_graph().draw_mermaid_png()
# #     with open("multi_agent_workflow.png", "wb") as f:
# #         f.write(image_data)
# #     print("✅ 工作流图已成功保存为当前目录下的 'multi_agent_workflow.png'")
# # except Exception as e:
# #     print(f"⚠️ 生成图片失败 ({e})。")
# #     print("👇 你可以复制以下 Mermaid 代码到 https://mermaid.live/ 网站上直接查看图表：")
# #     print("\n" + "="*40)
# #     print(graph.get_graph().draw_mermaid())
# #     print("="*40 + "\n")

# # ==========================================
# # 🏃 异步运行测试
# # ==========================================
# async def main():
#     config = {"configurable": {"thread_id": "distributed_run_01"}}
#     print("="*50)
#     print("🎬 启动微调 LLM 多智能体系统")
#     print("="*50)
    
#     # 1. 增加用户交互终端，收集核心参数（支持回车使用默认值，方便调试）
#     print("\n👉 请输入本次微调任务的配置：")
#     dataset_path = input("📂 数据集地址 (默认: /public/home/liqsh2/zly/medical_dataset_merged.json): ") or "/public/home/liqsh2/zly/medical_dataset_merged.json"
#     base_model_path = input("🧠 基座模型地址 (默认: /public/home/liqsh2/zly/base_model): ") or "/public/home/liqsh2/zly/base_model"
#     output_path = input("💾 模型保存地址 (默认: /public/home/liqsh2/zly/model): ") or "/public/home/liqsh2/zly/model"
#     task_desc = input("📝 任务描述/特殊要求 (默认: 帮我微调一个生物垂域LLM): ") or "帮我微调一个生物垂域LLM"
#     small_model_path = input("🔍 数据筛选小模型地址 (默认: /public/home/liqsh2/zly/base_model): ") or "/public/home/liqsh2/zly/base_model"

#     # 2. 将用户输入组装成结构化的初始 Prompt
#     initial_prompt = f"""
#     请帮我执行一个完整的模型微调自动化流程。以下是具体的环境配置参数：

#     - 任务描述: {task_desc}
#     - 数据筛选小模型路径: {small_model_path}
#     - 数据集路径: {dataset_path}
#     - 基座模型路径: {base_model_path}
#     - 输出/保存路径: {output_path}

#     请严格按照提供的路径参数，依次完成数据筛选、配置解析、启动微调以及最终的模型评估。
#     """
    
#     print("\n🚀 正在将配置下发给多智能体工作流，开始执行...")
    
#     # 3. 将组装好的 prompt 作为初始 HumanMessage 传入图
#     async for event in graph.astream(
#         {"messages": [HumanMessage(content=initial_prompt)]},
#         config,
#         stream_mode="values",
#     ):
#         pass

#     print("\n🎉 全部任务执行完毕！")

# if __name__ == "__main__":
#     asyncio.run(main())