import os, json
from dotenv import load_dotenv
from tools import try_calculator
import rag_simple as rag

# Carga configuración desde .env (proveedor, modelos, API keys, etc.)
load_dotenv()
PROVIDER = os.getenv("PROVIDER", "openai").lower()

# Inicializa cliente de modelo según proveedor elegido
openai_client = None
ollama_client = None
if PROVIDER == "openai":
    try:
        from openai import OpenAI
        openai_client = OpenAI()
    except Exception as e:
        print("[ADVERTENCIA] No se pudo inicializar OpenAI. ¿Instalaste 'openai' y seteaste OPENAI_API_KEY en .env?")
elif PROVIDER == "ollama":
    try:
        import ollama
        ollama_client = ollama
    except Exception as e:
        print("[ADVERTENCIA] No se pudo inicializar Ollama. ¿Instalaste Ollama y descargaste un modelo (p. ej. 'ollama run llama3.1')?")

MEM_PATH = "memory.json"

def load_memory():
    """Lee el historial de chat del archivo JSON (si existe)."""
    if os.path.exists(MEM_PATH):
        try:
            return json.load(open(MEM_PATH, "r", encoding="utf-8"))
        except Exception:
            pass
    return []

def save_memory(messages):
    """Guarda el historial de chat en disco."""
    json.dump(messages, open(MEM_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def llm_chat(messages, system_prompt):
    """Envía el historial al LLM elegido y devuelve la respuesta de texto."""
    if PROVIDER == "openai" and openai_client is not None:
        resp = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0.4,
        )
        return resp.choices[0].message.content
    elif PROVIDER == "ollama" and ollama_client is not None:
        resp = ollama_client.chat(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            messages=[{"role": "system", "content": system_prompt}] + messages
        )
        return resp["message"]["content"]
    return "Proveedor no disponible. Revisá tu .env y dependencias."

def format_with_context(user_input, rag_on):
    """
    Adjunta contexto de RAG (búsqueda en data/docs) si está activado.
    Citará fragmentos relevantes si los encuentra.
    """
    if not rag_on:
        return user_input
    hits = rag.top_k(user_input, k=4)
    if not hits:
        return user_input + "\n\n[No se hallaron docs locales]"
    ctx = "\n\n".join([f"[{i+1}] {doc_id}: {text[:200]}" for i, (doc_id, text, score) in enumerate(hits)])
    instruct = "Usa la información de CONTEXTO si es relevante. Si citas, referencia [n] y archivo."
    return f"{instruct}\n\nCONTEXTO:\n{ctx}\n\nPREGUNTA:\n{user_input}"

def main():
    print("=== Mi IA (CLI) ===")
    print("Comandos: ':reset', ':rag on', ':rag off', 'calc: 2+2'")
    messages = load_memory()
    rag_on = False

    system_prompt = (
        "Sos un asistente útil, directo y práctico. "
        "Cuando uses CONTEXTO citado, referencia [n] y el archivo. "
        "Si algo no sabés, decilo y proponé cómo averiguarlo."
    )

    while True:
        try:
            user = input("\nTú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego!")
            break

        if not user:
            continue

        # Comandos básicos
        if user == ":reset":
            messages = []
            save_memory(messages)
            print("Memoria borrada.")
            continue
        if user == ":rag on":
            rag_on = True
            print("RAG activado (leerá data/docs).")
            continue
        if user == ":rag off":
            rag_on = False
            print("RAG desactivado.")
            continue

        # Herramienta calculadora
        calc = try_calculator(user)
        if calc is not None:
            print(f"IA (tool): {calc}")
            # Podés elegir guardar también este intercambio en memoria si querés
            # messages.append({"role":"user","content":user})
            # messages.append({"role":"assistant","content":calc})
            # save_memory(messages)
            continue

        # Prepara el mensaje (con o sin contexto de RAG)
        content = format_with_context(user, rag_on)
        messages.append({"role": "user", "content": content})

        reply = llm_chat(messages, system_prompt)
        print("IA:", reply)
        messages.append({"role": "assistant", "content": reply})
        save_memory(messages)

if __name__ == "__main__":
    main()
