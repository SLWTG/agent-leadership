import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()
llm = ChatOpenAI(
    model=os.getenv('MODEL', 'deepseek-flash'),
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url=os.getenv('OPENAI_BASE_URL', 'https://api.deepseek.com'),
    temperature=0,
)

# 工具 1：沿用 P1
def get_stock_change(code: str) -> str:
    if not (isinstance(code, str) and code.isdigit() and len(code) == 6):
        return '股票代码应为6位数字'
    data = {'600000': ('浦发银行', 1.25), '000001': ('平安银行', -0.86), '601318': ('中国平安', 2.10)}
    row = data.get(code)
    return '未收录该股票代码' if not row else f'{row[0]}今日上涨 {row[1]}%'

# 工具 2：新增（加工具 = 加一个函数 + 注册一行）
def get_usd_cny() -> str:
    return '今日美元兑人民币汇率 7.05（示例数据）'

TOOL_HANDLERS = {'get_stock_change': get_stock_change, 'get_usd_cny': get_usd_cny}
llm_with_tools = llm.bind_tools(list(TOOL_HANDLERS.values()))

SYSTEM_PROMPT = """你是金融查询助手。你的目标：用工具查询行情与汇率，给出简洁准确回答。
【执行规则】1. 查涨跌必须调用 get_stock_change，禁止编造数字。2. 查汇率必须调用 get_usd_cny。3. 只回答金融相关问题，其他问题礼貌拒绝。"""

def execute_tool(name: str, args: dict):
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return f'未知工具: {name}'
    try:
        return handler(**args)
    except Exception as e:
        return f'工具执行失败: {e}'

def chat_loop(user_input: str, max_turns: int = 3):
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(user_input)]
    for _ in range(max_turns):
        try:
            resp = llm_with_tools.invoke(messages)
        except Exception as e:
            print(f'[API错误] {e}')
            return
        messages.append(resp)
        if not resp.tool_calls:
            print(f'[最终回答] {resp.content}')
            return
        for tc in resp.tool_calls:
            out = execute_tool(tc['name'], tc['args'])
            print(f'[工具调用] {tc["name"]}({tc["args"]}) -> {out}')
            messages.append(ToolMessage(out, tool_call_id=tc['id']))

if __name__ == '__main__':
    chat_loop(input('提问：').strip())
