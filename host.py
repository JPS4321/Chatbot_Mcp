import os, json, yaml, asyncio
from typing import Dict, Any

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

# ---- Carga API Key ----
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("Falta OPENAI_API_KEY en .env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------- Prompt de ruteo genérico (agnóstico de dominio) ----------
TOOL_SELECTION_SYSTEM = """Eres un orquestador de herramientas MCP.
Recibirás una lista de tools disponibles (nombre, descripción y schema JSON).
Tu tarea es: Dado un input del usuario, seleccionar UNA sola tool y construir un objeto JSON de argumentos
que respete estrictamente el schema (tipos y nombres de propiedades).

Reglas:
- Elige la tool cuya descripción/nombre mejor coincida con la intención del usuario.
- Construye "arguments" SOLO con propiedades permitidas por el inputSchema de la tool elegida.
- Casos ambiguos: si hay más de una tool plausible o faltan datos, usa "tool_name": null y "arguments": {}.
- No inventes propiedades ni cambies tipos: si el schema pide integer, no pases string.

Responde EXCLUSIVAMENTE con JSON:
{ "tool_name": string|null, "arguments": object, "reasoning_summary": string }
La propiedad "reasoning_summary" debe ser breve (máximo 2 oraciones).
"""

def build_tools_catalog(tools_resp) -> str:
    """Convierte la lista de tools del servidor actual en un bloque de texto que incluya nombre, descripción y schema."""
    lines = []
    for t in tools_resp.tools:
        schema_str = json.dumps(t.inputSchema, ensure_ascii=False)
        lines.append(f"- name: {t.name}\n  desc: {t.description}\n  schema: {schema_str}")
    return "\n".join(lines)

def ask_model_for_tool(user_message: str, tools_catalog: str) -> Dict[str, Any]:
    """Pide al modelo que elija una tool con argumentos válidos (agnóstico del dominio)."""
    prompt = f"""Herramientas disponibles:
{tools_catalog}

Usuario: {user_message}

Devuelve SOLO un JSON con: tool_name, arguments, reasoning_summary.
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
        return {"tool_name": None, "arguments": {}, "reasoning_summary": "No se pudo parsear la selección."}


def ask_model_for_final_answer(tool_output_text: str) -> str:
    """
    Toma directamente el output de la tool y lo reescribe en un texto entendible,
    sin agregar información externa.
    """
    system = "Eres un asistente que transforma salidas técnicas de tools en explicaciones claras y concisas en español. IMPORTANTE: No inventes ni agregues información externa, usa solamente lo que aparece en la salida de la tool."
    prompt = f"""Salida de la tool:
{tool_output_text}

Reescribe la información anterior en forma de explicación para el usuario.
No inventes nada adicional, limita tu respuesta solo a lo que aparece aquí.
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





async def main():
    # 1) Cargar config de servers.yaml 
    cfg = yaml.safe_load(open(CONFIG_FILE, "r", encoding="utf-8"))
    if "servers" not in cfg or SERVER_KEY not in cfg["servers"]:
        raise SystemExit(f"No se encontró la clave '{SERVER_KEY}' en {CONFIG_FILE}")
    s = cfg["servers"][SERVER_KEY]
    cmd, args, env = s["command"], s.get("args", []), s.get("env", {})

    # 2) Conectar con el servidor MCP 
    server_params = StdioServerParameters(command=cmd, args=args, env=env)
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("🤝 Conectado a tu servidor MCP (via stdio).")

            # 3) Descubrir tools del server (para ruteo)
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            tools_catalog = build_tools_catalog(tools)
            print("🛠️ Tools disponibles:", ", ".join(tool_names) or "(ninguna)")
            print("Comandos: tools | exit")

            # 4) Loop interactivo
            ps = PromptSession()
            while True:
                user_msg = (await ps.prompt_async("> ")).strip()
                if not user_msg:
                    continue
                if user_msg in ("exit", "quit"):
                    print("👋 Adiós")
                    break
                if user_msg == "tools":
                    print(tools_catalog)
                    continue

                # === Paso a paso smart routing ===
                print("\n=== PASO A PASO ===")
                print(f"1) Input:\n   {user_msg}")

                selection = ask_model_for_tool(user_msg, tools_catalog)
                tool_name = selection.get("tool_name")
                tool_args = selection.get("arguments", {})
                thinking = selection.get("reasoning_summary", "")

                print("2) Pensando (modelo → selección):")
                print(f"   reasoning_summary: {thinking}")
                print(f"   tool_name: {tool_name}")
                print(f"   arguments: {json.dumps(tool_args, ensure_ascii=False)}")

                # 5) Llamar tool si aplica
                tool_output_text = ""
                if tool_name and tool_name in tool_names:
                    try:
                        result = await session.call_tool(name=tool_name, arguments=tool_args)
                        collected = []
                        for c in result.content:
                            if c.type == "text":
                                txt = c.text
                                try:
                                    txt = json.dumps(json.loads(txt), ensure_ascii=False, indent=2)
                                except Exception:
                                    pass
                                collected.append(txt)
                            else:
                                collected.append(f"[{c.type}]")
                        tool_output_text = "\n".join(collected).strip()
                    except Exception as e:
                        tool_output_text = f"[ERROR al llamar {tool_name}: {e}]"
                else:
                    tool_output_text = "(No se seleccionó tool válida; respuesta directa del modelo)."

                print("3) Output de la tool:")
                print(f"   {tool_output_text}")

                # 6) Respuesta final
                final_answer = ask_model_for_final_answer(tool_output_text)
                print("4) Respuesta final:")
                print(final_answer)
                print("====================\n")

if __name__ == "__main__":
    asyncio.run(main())
