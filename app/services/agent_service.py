from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.tools import tool

from app.services.llm_service import get_llm
from app.core.prompts import AGENT_SYSTEM_PROMPT

# ================= 1. 定义你的工具 =================
@tool
def get_weather(location: str) -> str:
    """获取指定城市的天气情况。"""
    return f"{location}今天天气晴朗，气温 25 度。"

@tool
def calculate_salary(gross_salary: float) -> float:
    """计算税后工资，输入为税前工资。"""
    return gross_salary * 0.8

tools = [get_weather, calculate_salary]

# ================= 2. 初始化基础 Agent =================
llm = get_llm()

# 将普通字符串组装成 LangChain 标准的对话模板 (解决之前报错的核心)
prompt = ChatPromptTemplate.from_messages([
    ("system", AGENT_SYSTEM_PROMPT), 
    MessagesPlaceholder(variable_name="chat_history"),     # 预留记忆空位
    ("user", "{input}"),                                   # 预留用户输入空位
    MessagesPlaceholder(variable_name="agent_scratchpad"), # 预留 Agent 思考草稿本空位
])

# 将大模型、工具和【组装好的 prompt 对象】组合成 Agent
agent = create_tool_calling_agent(llm, tools, prompt)
# 创建执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ================= 3. 会话记忆存储配置 =================
# 模拟数据库，用来存不同用户的聊天记录 (生产环境会用 Redis)
session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """根据 session_id 获取对应的聊天记录"""
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# 给 Agent 穿上“记忆装甲”
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",        # 用户输入对应的 key
    history_messages_key="chat_history" # 历史记录存放在 prompt 中的 key
)

# ================= 4. 暴露给外部的主函数 =================
def process_agent_query(query: str, session_id: str) -> str:
    """
    处理 Agent 查询，带记忆功能
    """
    response = agent_with_chat_history.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}} # 传入当前用户的 ID
    )
    return response["output"]