import os, json, yaml, asyncio, re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from openai import OpenAI

# MCP client
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession

CONFIG_FILE = "servers.yaml"
SERVER_KEY = "remote_echo"  
OPENAI_MODEL = "gpt-4o-mini"

# Load API Key
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("Missing OPENAI_API_KEY in .env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Context and log files
CONTEXT_FILE = "chat_context.json"
LOG_FILE = "chatbot_log.txt"

# Reset context file at startup
with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        {"server": SERVER_KEY, "history": [], "last_tool_memory": {}, "last_list": None},
        f, ensure_ascii=False, indent=2
    )

conversation_history: List[Dict[str, str]] = []

# Herramientas de contexto
def save_to_context(entry: Dict[str, Any]):
    with open(CONTEXT_FILE, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["history"].append(entry)
        # Recordar ultima tool utilizada
        if entry.get("tool_used") and entry.get("arguments"):
            data["last_tool_memory"][entry["tool_used"]] = entry["arguments"]
        # Guardar lista si el output fue ese
        if entry.get("last_list"):
            data["last_list"] = entry["last_list"]
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()
# Devuelve los ultimos argumentos en caso se usaron con una herramienta especifica
def get_last_args_for_tool(tool_name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not tool_name:
        return None
    try:
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["last_tool_memory"].get(tool_name)
    except FileNotFoundError:
        return None
#Recupera la ultima lista usada (Esto para cosas como el primero, segundo,etc)
def get_last_list() -> Optional[Dict[str, Any]]:
    try:
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("last_list")
    except FileNotFoundError:
        return None
#Guardar cada interaccion en un archivo .txt
def log_interaction(entry: Dict[str, Any]):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

#  Tool selection prompt  (En ingles porque mi base de datos esta en ingles :D)
TOOL_SELECTION_SYSTEM = """You are an MCP tool orchestrator.
You will receive a list of available tools (name, description and JSON schema).
Your task: Given a user input, select ONE tool and construct a JSON object of arguments
that strictly follows the schema.

Rules:
- Pick the tool whose description/name best matches the user's intent.
- If the question does NOT match any tool, use "tool_name": null.
- Respond ONLY with JSON:
{ "tool_name": string|null, "arguments": object, "reasoning_summary": string }
"""
#String de herramientas expuestas disponibles
def build_tools_catalog(tools_resp) -> str:
    lines = []
    for t in tools_resp.tools:
        schema_str = json.dumps(t.inputSchema, ensure_ascii=False)
        lines.append(f"- name: {t.name}\n  desc: {t.description}\n  schema: {schema_str}")
    return "\n".join(lines)

# Crea un diccionario que mapea el nombre de cada herramienta con su esquema JSON de argumentos.
#Se usa para saber qué campos esperan las herramientas.
def index_tool_schemas(tools_resp) -> Dict[str, Dict[str, Any]]:
    return {t.name: t.inputSchema for t in tools_resp.tools}

#Pregutna al modelo que herramienta deberia usar y le dal el mensaje del usuario
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
#Pide al modelo que transforme la salida de una herramienta en un texto tipo parrafos.
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
#Se usa cuando no hay ninguna herramienta adecuada. Siemplemente dice que no hay una tool y da una respuesta generica

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

# ---------------------- Ordinal & entity helpers ----------------------
ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10
}

ORDINAL_PATTERNS = [
    re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.I),          # 1st, 2nd, 3rd, 4th
    re.compile(r"\bnumber\s+(\d+)\b", re.I),              # number 3
    re.compile(r"\bno\.\s*(\d+)\b", re.I),                # no. 2
    re.compile(r"(?:^|#)\s*(\d+)\b"),                     # #1 or leading 1
]
# Detecta si un texto contiene un  y devuelve el índice cero-based (0 para first, 1 para second, etc.)
def parse_ordinal_index(text: str) -> Optional[int]:
    """Return zero-based index if an ordinal is detected, else None."""
    low = text.lower()
    for w, n in ORDINAL_WORDS.items():
        # accept "first", "first one", "first game"
        if re.search(rf"\b{w}\b", low):
            return n - 1
    for pat in ORDINAL_PATTERNS:
        m = pat.search(low)
        if m:
            try:
                return int(m.group(1)) - 1
            except Exception:
                continue
    return None

def pick_entity_key(items: List[Dict[str, Any]]) -> Optional[str]:
    """Pick a reasonable key to represent the entity label in a list of dicts."""
    if not items:
        return None
    candidates = ["Name", "name", "title", "Title", "id", "ID"]
    for key in candidates:
        if all(isinstance(it, dict) and key in it for it in items):
            return key
    # fallback: first string-looking field across items
    for k in items[0].keys():
        if all(isinstance(it.get(k), (str, int, float)) for it in items):
            return k
    return None
#Intenta interpretar la salida de una herramienta como JSON con un campo results.
def extract_list_entities_from_tool_output(tool_output_text: str) -> Optional[Dict[str, Any]]:
    """If tool output looks like JSON with a 'results' list, extract representative labels."""
    try:
        obj = json.loads(tool_output_text)
    except Exception:
        return None
    results = obj.get("results")
    if not isinstance(results, list) or not results:
        return None
    if isinstance(results[0], dict):
        key = pick_entity_key(results)
        if not key:
            return None
        labels = []
        for it in results:
            v = it.get(key)
            labels.append(str(v) if v is not None else "")
        return {"entity_key": key, "labels": labels}
    # list of scalars
    labels = [str(x) for x in results]
    return {"entity_key": None, "labels": labels}
#Busca dentro del esquema de argumentos de una herramienta el campo preferido para inyectar valores de entidades (name, id, title).
def preferred_arg_key_for_tool(schema: Dict[str, Any]) -> Optional[str]:
    """Choose a preferred argument name to fill (name > id > title)."""
    try:
        props = schema.get("properties", {})
        for cand in ("name", "id", "title"):
            if cand in props:
                return cand
    except Exception:
        pass
    return None
#Si el usuario dice primero y asi, toma la lista más reciente
def resolve_ordinal_reference_in_args(
    user_msg: str,
    tool_name: Optional[str],
    tool_args: Dict[str, Any],
    tool_schemas: Dict[str, Dict[str, Any]],
    last_list: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """If user refers to 'first/2nd/#3', map to the corresponding label from the last list and
    inject it into the preferred argument key of the selected tool."""
    if not tool_name or tool_name not in tool_schemas or not last_list:
        return tool_args

    idx = parse_ordinal_index(user_msg)
    if idx is None:
        # Also handle explicit phrases like "the first game/item/one"
        if re.search(r"\b(top\s*1|the\s+first\s+(one|item|game))\b", user_msg, re.I):
            idx = 0
        else:
            return tool_args

    labels = last_list.get("labels") or []
    if not labels or idx < 0 or idx >= len(labels):
        return tool_args

    arg_key = preferred_arg_key_for_tool(tool_schemas[tool_name]) or "name"
    # Only override if arg is missing or looks like a placeholder (e.g., "first game")
    current_val = str(tool_args.get(arg_key, "")).strip().lower()
    if (not current_val) or re.fullmatch(r"(first|second|third|\d+(st|nd|rd|th)?|#?\d+|number\s+\d+)(\s+\w+)?", current_val):
        tool_args[arg_key] = labels[idx]
    return tool_args

#  Main 
#Aqui es el flujo basico
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
            print(f" Connected to MCP server: {SERVER_KEY}")
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            tools_catalog = build_tools_catalog(tools)
            tool_schemas = index_tool_schemas(tools)
            print("🛠️ Available tools:", ", ".join(tool_names) or "(none)")
            print("Commands: tools | context | exit")

            ps = PromptSession()
            while True:
                user_msg = (await ps.prompt_async("> ")).strip()
                if not user_msg:
                    continue
                if user_msg in ("exit", "quit"):
                    print(" Goodbye")
                    break
                if user_msg == "tools":
                    print(tools_catalog)
                    continue
                if user_msg == "context":
                    try:
                        print(open(CONTEXT_FILE, "r", encoding="utf-8").read())
                    except Exception as e:
                        print(f"[ERROR reading context: {e}]")
                    continue

                selection = ask_model_for_tool(user_msg, tools_catalog)
                tool_name = selection.get("tool_name")
                tool_args = selection.get("arguments", {}) or {}
                last_args = get_last_args_for_tool(tool_name)
                if tool_name and last_args and not tool_args:
                    tool_args = last_args

                # Resolve ordinal references like "the first one" using last list
                tool_args = resolve_ordinal_reference_in_args(
                    user_msg=user_msg,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_schemas=tool_schemas,
                    last_list=get_last_list(),
                )

                print("\n=== STEP-BY-STEP ===")
                print(f"1) Input:\n   {user_msg}")
                print("2) Thinking:")
                print(f"   reasoning_summary: {selection.get('reasoning_summary')}")
                print(f"   tool_name: {tool_name}")
                print(f"   arguments: {json.dumps(tool_args, ensure_ascii=False)}")

                tool_output_text = ""
                extracted_list = None
                if tool_name and tool_name in tool_names:
                    try:
                        result = await session.call_tool(name=tool_name, arguments=tool_args)
                        collected = [c.text for c in result.content if c.type == "text"]
                        tool_output_text = "\n".join(collected).strip()
                        final_answer = ask_model_for_final_answer(tool_output_text)

                        # Try to extract a labeled list from tool output (to support follow-ups like "the first one")
                        extracted_list = extract_list_entities_from_tool_output(tool_output_text)
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
                if extracted_list:
                    entry["last_list"] = {
                        "from_tool": tool_name,
                        "entity_key": extracted_list.get("entity_key"),
                        "labels": extracted_list.get("labels"),
                    }
                save_to_context(entry)
                log_interaction(entry)

if __name__ == "__main__":
    asyncio.run(main())
