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