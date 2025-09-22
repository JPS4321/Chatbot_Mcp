import os, json, yaml, asyncio
from typing import Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from openai import OpenAI

# MCP client
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession

CONFIG_FILE = "servers.yaml"
SERVER_KEY = "game_stats"  
OPENAI_MODEL = "gpt-4o-mini"

# ---- Load API Key ----
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("Missing OPENAI_API_KEY in .env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Context and log files
CONTEXT_FILE = "chat_context.json"
LOG_FILE = "chatbot_log.txt"

# Reset context file at startup
with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
    json.dump({"server": SERVER_KEY, "history": [], "last_tool_memory": {}}, f, ensure_ascii=False, indent=2)

conversation_history = []

# -------- Context Management --------
def save_to_context(entry: Dict[str, Any]):
    with open(CONTEXT_FILE, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["history"].append(entry)
        if entry.get("tool_used") and entry.get("arguments"):
            data["last_tool_memory"][entry["tool_used"]] = entry["arguments"]
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()

def get_last_args_for_tool(tool_name: str) -> Dict[str, Any] | None:
    try:
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["last_tool_memory"].get(tool_name)
    except FileNotFoundError:
        return None

def log_interaction(entry: Dict[str, Any]):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# -------- Tool Selection System Prompt --------
TOOL_SELECTION_SYSTEM = """You are an MCP tool orchestrator.
You will receive a list of available tools (name, description and JSON schema).
Your task is: Given a user input, select ONE single tool and build a JSON object with arguments
that strictly follows the schema.

Rules:
- Pick the tool whose description/name best matches the user's intent.
- If the question does NOT match any tool, use "tool_name": null.
- Respond ONLY with JSON:
{ "tool_name": string|null, "arguments": object, "reasoning_summary": string }
"""

def build_tools_catalog(tools_resp) -> str:
    lines = []
    for t in tools_resp.tools:
        schema_str = json.dumps(t.inputSchema, ensure_ascii=False)
        lines.append(f"- name: {t.name}\n  desc: {t.description}\n  schema: {schema_str}")
    return "\n".join(lines)

def ask_model_for_tool(user_message: str, tools_catalog: str) -> Dict[str, Any]:
    prompt = f"""Available tools:
{tools_catalog}

User message: {user_message}

Return ONLY a JSON with: tool_name, arguments, reasoning_summary.
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": TOOL_SELECTION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except Exception:
        return {"tool_name": None, "arguments": {}, "reasoning_summary": "Could not parse model output."}

def ask_model_for_final_answer(tool_output_text: str) -> str:
    system = "Turn tool outputs into clear, concise explanations in English. Use ONLY the information in the output."
    prompt = f"""Tool output:
{tool_output_text}

Rewrite this information in a user-friendly way.
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

def ask_model_basic_fallback(user_message: str) -> str:
    system = (
        "Reply in English with a VERY BASIC answer (max 2 sentences). "
        "Do not invent specific details. "
        "Always start with: 'No tool available for this. Basic answer:'"
    )
    messages = [{"role": "system", "content": system}] + conversation_history + [
        {"role": "user", "content": user_message}
    ]
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.0,
    )
    reply = resp.choices[0].message.content.strip()
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

# ---------------- Main ----------------
async def main():
    cfg = yaml.safe_load(open(CONFIG_FILE, "r", encoding="utf-8"))
    if "servers" not in cfg or SERVER_KEY not in cfg["servers"]:
        raise SystemExit(f"Server key '{SERVER_KEY}' not found in {CONFIG_FILE}")
    s = cfg["servers"][SERVER_KEY]
    cmd, args, env = s["command"], s.get("args", []), s.get("env", {})

    server_params = StdioServerParameters(command=cmd, args=args, env=env)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print(f"🤝 Connected to MCP server: {SERVER_KEY}")
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            tools_catalog = build_tools_catalog(tools)
            print("🛠️ Available tools:", ", ".join(tool_names) or "(none)")
            print("Commands: tools | exit")

            ps = PromptSession()
            while True:
                user_msg = (await ps.prompt_async("> ")).strip()
                if not user_msg:
                    continue
                if user_msg in ("exit", "quit"):
                    print("👋 Goodbye")
                    break
                if user_msg == "tools":
                    print(tools_catalog)
                    continue

                selection = ask_model_for_tool(user_msg, tools_catalog)
                tool_name = selection.get("tool_name")
                tool_args = selection.get("arguments", {})
                last_args = get_last_args_for_tool(tool_name) if tool_name else None
                if tool_name and last_args and not tool_args:
                    tool_args = last_args

                print("\n=== STEP-BY-STEP ===")
                print(f"1) Input:\n   {user_msg}")
                print("2) Thinking:")
                print(f"   reasoning_summary: {selection.get('reasoning_summary')}")
                print(f"   tool_name: {tool_name}")
                print(f"   arguments: {json.dumps(tool_args, ensure_ascii=False)}")

                tool_output_text = ""
                if tool_name and tool_name in tool_names:
                    try:
                        result = await session.call_tool(name=tool_name, arguments=tool_args)
                        collected = [c.text for c in result.content if c.type == "text"]
                        tool_output_text = "\n".join(collected).strip()
                        final_answer = ask_model_for_final_answer(tool_output_text)
                    except Exception as e:
                        final_answer = f"[ERROR calling {tool_name}: {e}]"
                else:
                    print("   (No tool) No tool matched this query.")
                    final_answer = ask_model_basic_fallback(user_msg)

                print("3) Final Answer:")
                print(final_answer)
                print("====================\n")

                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "user": user_msg,
                    "tool_used": tool_name,
                    "arguments": tool_args,
                    "tool_output": tool_output_text,
                    "final_answer": final_answer,
                }
                save_to_context(entry)
                log_interaction(entry)

if __name__ == "__main__":
    asyncio.run(main())
