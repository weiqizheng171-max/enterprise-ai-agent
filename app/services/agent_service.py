from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.services.llm_service import get_llm
from app.core.prompts import AGENT_SYSTEM_PROMPT

# 定义工具 1：模拟查天气
@tool
def get_weather(location: str) -> str:
    """获取指定城市的天气情况。"""
    # 模拟 API 调用
    return f"{location}今天天气晴朗，气温25度，适合出行。"

# 定义工具 2：模拟算薪资
@tool
def calculate_net_salary(gross_salary: float) -> str:
    """计算税后薪资。输入税前薪资。"""
    net = gross_salary * 0.8  # 简单模拟扣税20%
    return f"税前薪资 {gross_salary} 元，预计税后到手约为 {net} 元。"

tools = [get_weather, calculate_net_salary]

def process_agent_query(query: str) -> str:
    llm = get_llm()
    
    # 构建 Agent 提示词模板 (使用 MessagesPlaceholder 兼容性更强)
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 使用 create_openai_tools_agent，完美兼容 DeepSeek 的工具调用
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    result = agent_executor.invoke({"input": query})
    return result["output"]