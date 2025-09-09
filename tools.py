import re
from typing import Optional

def try_calculator(message: str) -> Optional[str]:
    """
    Activa la calculadora cuando el mensaje empieza con 'calc:'.
    Ejemplos:
      - calc: 2+2*3
      - calc: (12 + 8) / 5
    Solo permite números, + - * / y paréntesis.
    """
    if not message.lower().strip().startswith("calc:"):
        return None

    expr = message.split(":", 1)[1].strip()

    # Validación simple de seguridad (whitelist de caracteres)
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr):
        return "Formato inválido para calc. Solo números y + - * / ( )."

    try:
        # Eval “seguro”: sin builtins ni nombres disponibles
        result = eval(expr, {"__builtins__": {}}, {})
        return f"Resultado: {result}"
    except Exception as e:
        return f"Error al calcular: {e}"
