import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com"),
)

# —— 八段式 system（这就是模板的落地示范）——
SYSTEM_PROMPT = """你是A股行情查询助手，服务于个人投资者。你的目标：用工具查询个股当日涨跌幅，给出简洁准确回答。

【输入说明】你会收到用户的自然语言提问；若缺少6位股票代码，先请用户提供，不猜测。

【输出规范】一句话回答，包含股票名称与涨跌幅，简体中文。

【执行规则】1. 查涨跌必须调用 get_stock_change 工具，禁止编造数字。2. 工具返回 found=false 时如实告知"未收录"。3. 只回答行情相关问题，其他问题礼貌拒绝。

【工具使用】工具：get_stock_change(code)；调用时机：用户询问涨跌且已有代码时；代码未知先索要；工具失败如实说明。

【质量自检】输出前检查：①数字来自工具返回 ②含股票名称 ③未编造未收录数据。"""

# —— 假数据（生产时换成真实接口，P2 再做）——
STOCKS = {
    "600000": {"name": "浦发银行", "change_pct": 1.25},
    "000001": {"name": "平安银行", "change_pct": -0.86},
    "601318": {"name": "中国平安", "change_pct": 2.10},
}

def get_stock_change(code: str) -> dict:
    s = STOCKS.get(code)
    if not s:
        return {"code": code, "found": False, "message": "未收录该股票代码"}
    return {"code": code, "name": s["name"], "change_pct": s["change_pct"], "found": True}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_stock_change",
        "description": "查询A股个股当日涨跌幅。用户询问某只股票涨跌时调用，需传入6位股票代码。",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "6位A股股票代码，如600000"}},
            "required": ["code"],
        },
    },
}]

def execute_tool(name: str, arguments: str) -> dict:
    args = json.loads(arguments)
    if name == "get_stock_change":
        return get_stock_change(args["code"])
    raise ValueError(f"未知工具: {name}")

def chat_loop(user_input: str, max_turns: int = 3):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    for _ in range(max_turns):
        resp = client.chat.completions.create(
            model=os.getenv("MODEL", "deepseek-chat"),
            messages=messages,
            tools=TOOLS,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump())

        if msg.tool_calls:
            for tc in msg.tool_calls:
                result = execute_tool(tc.function.name, tc.function.arguments)
                print(f"[工具调用] {tc.function.name}({tc.function.arguments}) -> {result}")
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, ensure_ascii=False)})
            continue  # 回填后再请求一轮，模型给出最终回答

        print(f"[最终回答] {msg.content}")
        return
    print("达到最大轮次，停止（防死循环 = A5 的第一课）")

if __name__ == "__main__":
    chat_loop(input("提问：").strip())
