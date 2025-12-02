#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DUDU evaluation script, parametrizable.

- Lee un CSV de resultados de RAG (resultados_emails_*.csv),
  que debe tener al menos las columnas:
    - question
    - gt_answer
    - pred_answer
    - retrieved_chunks   (texto que usas como contexto R)
- Para cada fila llama a Gemini (gemini-2.5-flash) como "juez"
  y devuelve E, C, H + justification en el mismo idioma de la pregunta.
- Calcula un DUDU_score sencillo por fila.
- Guarda todo en otro CSV (input + columnas DUDU_*).

Además:
- Incluye espera entre llamadas para no pasar el rate limit.
- Reintenta una vez si hay error 429 / RESOURCE_EXHAUSTED.
"""

import os
import csv
import json
import time
from statistics import mean
from typing import Optional, Dict, Any, List

# ================== GEMINI CLIENT ==================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Si no tienes python-dotenv, no pasa nada: usa solo variables de entorno del sistema
    pass

from google import genai
from google.genai import types as genai_types


MODEL_NAME = "gemini-2.5-flash"

# límite de caracteres para el contexto pasado al LLM
MAX_CONTEXT_CHARS = 2000
# separador entre llamadas para no romper rate limit
TIME_BETWEEN_CALLS = 6.0
# segundos a dormir si nos devuelve 429/RESOURCE_EXHAUSTED
SLEEP_ON_QUOTA_ERROR = 30.0


# ================== PROMPTS ==================

PROMPT_HEADER_ES = """
Actúas como un evaluador experto de sistemas de Preguntas y Respuestas sobre documentos (DocVQA / RAG).

Tu tarea es comparar:
- La pregunta del usuario (Q),
- La respuesta correcta de referencia (GT),
- La respuesta generada por el sistema (PRED),
- El contexto textual recuperado (R).

Debes asignar tres puntuaciones enteras, cada una entre 0 y 3:

E (Exactitud semántica respecto a GT):
- 0 = completamente incorrecta o contradictoria.
- 1 = contiene algo relevante pero con errores importantes.
- 2 = mayormente correcta, con pequeños errores u omisiones.
- 3 = esencialmente correcta y completa respecto a GT.

C (Cobertura / completitud de la información relevante):
- 0 = ignora casi toda la información clave.
- 1 = cubre una parte pequeña de lo importante.
- 2 = cubre la mayor parte de la información clave.
- 3 = cubre prácticamente toda la información relevante.

H (Uso de contexto / Alucinación respecto a R):
- 0 = fuerte alucinación, inventa detalles que no aparecen en GT ni en R.
- 1 = mezcla información correcta con detalles dudosos o inventados.
- 2 = mayormente apoyada por R, con alguna duda menor.
- 3 = no hay alucinación, todo se apoya claramente en R y/o GT.

Instrucciones importantes:
- Usa solo la información contenida en GT y R. No inventes hechos externos.
- Si PRED contradice claramente GT o R, baja E y H.
- La justificación debe ser breve (2–4 líneas) y en español.
- Devuelve ÚNICAMENTE un JSON válido, sin texto extra, sin comentarios, sin explicaciones adicionales.

Formato EXACTO de salida (ejemplo):
{"E": 3, "C": 2, "H": 1, "justification": "explicación breve en español"}
"""

PROMPT_HEADER_EN = """
You act as an evaluation judge for document QA systems (DocVQA / RAG).

Your task is to compare:
- The user question (Q),
- The gold reference answer (GT),
- The model answer (PRED),
- The retrieved textual context (R).

You must assign three integer scores, each between 0 and 3:

E (Semantic Exactness w.r.t. GT):
- 0 = completely wrong or contradictory.
- 1 = somewhat related but with major mistakes.
- 2 = mostly correct with minor errors or omissions.
- 3 = essentially correct and complete w.r.t. GT.

C (Coverage / completeness of relevant information):
- 0 = ignores almost all key information.
- 1 = covers only a small part of what matters.
- 2 = covers most of the key information.
- 3 = covers practically all relevant details.

H (Use of context / Hallucination w.r.t. R):
- 0 = strong hallucination, invents details not in GT or R.
- 1 = mixes correct info with dubious or invented details.
- 2 = mostly grounded in R, with minor doubts.
- 3 = fully grounded in R and/or GT, no hallucination.

Important instructions:
- Use only information from GT and R. Do not introduce external facts.
- If PRED clearly contradicts GT or R, lower E and H.
- The justification must be short (2–4 sentences) and in English.
- Return ONLY a valid JSON object, with no extra text, no comments, no prose around it.

Exact output format (example):
{"E": 3, "C": 2, "H": 1, "justification": "short explanation in English"}
"""


# ================== UTILS ==================


def detect_language(question: str, gt_answer: str, pred_answer: str) -> str:
    """
    Heurística súper simple:
    - si vemos caracteres típicos del español, devolvemos 'es'
    - si no, 'en'
    """
    text = (question or "") + " " + (gt_answer or "") + " " + (pred_answer or "")
    spanish_chars = "áéíóúñÁÉÍÓÚÑ¡¿"
    if any(ch in text for ch in spanish_chars):
        return "es"
    return "en"


def truncate_context(raw: Optional[str], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if not raw:
        return "N/A"
    raw = str(raw)
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + "\n...\n[Context truncated]"


def build_prompt(
    question: str,
    gt_answer: str,
    pred_answer: str,
    retrieved_snippets: str,
    lang: str,
) -> str:
    header = PROMPT_HEADER_ES if lang == "es" else PROMPT_HEADER_EN

    prompt = (
        header
        + "\n\n=== INPUT ===\n"
        + f"Q (question): {question}\n"
        + f"GT (gold answer): {gt_answer}\n"
        + f"PRED (model answer): {pred_answer}\n"
        + "R (retrieved context):\n"
        + retrieved_snippets
        + "\n\n=== OUTPUT JSON ===\n"
        + 'Return ONLY a JSON object like {"E": 0, "C": 0, "H": 0, "justification": "..."}\n'
    )
    return prompt


def get_gemini_client() -> genai.Client:
    """
    Usa GEMINI_API_KEY o GOOGLE_API_KEY del entorno.
    """
    api_key_gemini = os.getenv("GEMINI_API_KEY")
    api_key_google = os.getenv("GOOGLE_API_KEY")

    api_key = None
    if api_key_gemini:
        print("Using GEMINI_API_KEY from environment/.env.")
        api_key = api_key_gemini
    if api_key_google:
        # Si tienes ambos, priorizamos GOOGLE_API_KEY (puedes cambiarlo si quieres).
        print("Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.")
        api_key = api_key_google

    if not api_key:
        raise RuntimeError(
            "No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or .env file."
        )

    client = genai.Client(api_key=api_key)
    return client


def _extract_text_from_response(resp: Any) -> str:
    """
    Extrae el texto (JSON en string) desde la respuesta de Gemini.
    Maneja tanto resp.text como resp.candidates[..].content.parts.
    """
    # Intento directo
    text = getattr(resp, "text", None)
    if text:
        return text

    # Fallback: mirar candidates/parts
    try:
        if resp.candidates:
            parts = resp.candidates[0].content.parts
            buf = []
            for p in parts:
                if hasattr(p, "text") and p.text:
                    buf.append(p.text)
            if buf:
                return "\n".join(buf)
    except Exception:
        pass

    # Si llegamos aquí, no hay texto visible
    raise RuntimeError(f"Empty response from Gemini (no text / no parts). Raw resp={resp}")


def _safe_json_loads(raw: str) -> Dict[str, Any]:
    """
    Intenta hacer json.loads con un par de heurísticas suaves
    (recortar antes/después de la primera/última llave).
    """
    raw = raw.strip()
    # Normalizamos: si hay basura fuera de { ... }
    if "{" in raw and "}" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        raw = raw[start:end].strip()

    return json.loads(raw)


def call_llm_judge(
    client: genai.Client,
    question: str,
    gt_answer: str,
    pred_answer: str,
    retrieved_chunks: str,
    max_retries: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    Llama al modelo de Gemini y devuelve un dict con E, C, H, justification.
    Si hay error de cuota (429/RESOURCE_EXHAUSTED) reintenta una vez
    después de dormir un rato. Si todo falla, devuelve None.
    """
    lang = detect_language(question, gt_answer, pred_answer)
    ctx = truncate_context(retrieved_chunks, MAX_CONTEXT_CHARS)
    prompt = build_prompt(question, gt_answer, pred_answer, ctx, lang)

    for attempt in range(1, max_retries + 1):
        try:
            config = genai_types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            )

            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )

            text = _extract_text_from_response(resp)
            data = _safe_json_loads(text)

            # Normalizamos campos esperados
            E = int(data.get("E", 0))
            C = int(data.get("C", 0))
            H = int(data.get("H", 0))
            just = str(data.get("justification", "")).strip()

            # Clampeamos por seguridad
            E = max(0, min(3, E))
            C = max(0, min(3, C))
            H = max(0, min(3, H))

            return {"E": E, "C": C, "H": H, "justification": just}

        except Exception as e:
            msg = str(e)
            print(f"⚠️ Error calling LLM judge (attempt {attempt}/{max_retries}): {msg}")

            is_quota = (
                "RESOURCE_EXHAUSTED" in msg
                or "quota" in msg.lower()
                or "429" in msg
            )

            if is_quota and attempt < max_retries:
                print("   Probable error de cuota, esperando un poco antes de reintentar...")
                time.sleep(SLEEP_ON_QUOTA_ERROR)
                continue

            # cualquier otro error, o ya sin reintentos
            return None


def compute_dudu_score(E: int, C: int, H: int) -> float:
    """
    Ejemplo sencillo de DUDU_score:
    - Normalizamos cada subscore a [0,1]
    - Pesos: 0.5 para E, 0.3 para C, 0.2 para H
    """
    e_n = E / 3.0
    c_n = C / 3.0
    h_n = H / 3.0
    return 0.5 * e_n + 0.3 * c_n + 0.2 * h_n


# ================== MAIN UTILS ==================


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calcula promedio de E, C, H, DUDU_score y algunos contadores.
    Devuelve un dict con resúmenes numéricos (útil para multi-eval).
    """
    def to_int(x, default=0):
        try:
            return int(x)
        except Exception:
            return default

    def to_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    E_vals = [to_int(r.get("E", 0)) for r in rows]
    C_vals = [to_int(r.get("C", 0)) for r in rows]
    H_vals = [to_int(r.get("H", 0)) for r in rows]
    dudu_scores = [to_float(r.get("DUDU_score", 0.0)) for r in rows]

    avg_E = mean(E_vals)
    avg_C = mean(C_vals)
    avg_H = mean(H_vals)
    avg_dudu = mean(dudu_scores)

    total = len(rows)
    num_exact = sum(1 for e in E_vals if e == 3)
    num_partial = sum(1 for e in E_vals if e in (1, 2))
    num_wrong = sum(1 for e in E_vals if e == 0)
    num_hallu = sum(1 for h in H_vals if h > 0)

    print("\n📊 DUDU SUMMARY (global)")
    print(f"- Total examples: {total}")
    print(f"- Avg E (Exactness): {avg_E:.3f}  (3=exact match)")
    print(f"- Avg C (Context coherence): {avg_C:.3f}")
    print(f"- Avg H (Hallucination severity): {avg_H:.3f}")
    print(f"- Avg DUDU_score: {avg_dudu:.4f}")
    print(f"- Exact answers (E=3): {num_exact}/{total}")
    print(f"- Partially correct (E=1 or 2): {num_partial}/{total}")
    print(f"- Incorrect (E=0): {num_wrong}/{total}")
    print(f"- With hallucination (H>0): {num_hallu}/{total}")

    return {
        "avg_E": avg_E,
        "avg_C": avg_C,
        "avg_H": avg_H,
        "avg_DUDU": avg_dudu,
        "num_exact": num_exact,
        "num_partial": num_partial,
        "num_wrong": num_wrong,
        "num_hallu": num_hallu,
        "total": total,
    }


def run_dudu_eval_for_file(
    input_csv: str,
    output_csv: str,
    max_rows: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evalúa un CSV concreto (input_csv) y guarda el resultado anotado en output_csv.
    Devuelve un dict con resumen de E, C, H, DUDU_score (para multi-eval).
    """
    client = get_gemini_client()

    # 1) Leer CSV de resultados del RAG
    rows = []
    with open(input_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            rows.append(row)
            if max_rows is not None and i >= max_rows:
                break

    print(f"✅ Loaded {len(rows)} rows from {input_csv}")

    out_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        q = row.get("question", "")
        gt = row.get("gt_answer", "")
        pred = row.get("pred_answer", "")
        retrieved = row.get("retrieved_chunks", "")

        print(f"\n➡️ DUDU eval [{idx}/{len(rows)}]")
        print(f"   Question: {q}")

        llm_result = call_llm_judge(client, q, gt, pred, retrieved)

        if llm_result is None:
            # dejamos columnas vacías si falló
            row["E"] = ""
            row["C"] = ""
            row["H"] = ""
            row["DUDU_score"] = ""
            row["dudu_justification"] = ""
        else:
            E = llm_result["E"]
            C = llm_result["C"]
            H = llm_result["H"]
            just = llm_result["justification"]

            score = compute_dudu_score(E, C, H)

            row["E"] = E
            row["C"] = C
            row["H"] = H
            row["DUDU_score"] = f"{score:.4f}"
            row["dudu_justification"] = just

        out_rows.append(row)

        # pequeña pausa entre llamadas para no exceder rate limit
        time.sleep(TIME_BETWEEN_CALLS)

    # 2) Guardar CSV enriquecido
    if out_rows:
        fieldnames = list(out_rows[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"💾 DUDU-annotated results saved to: {output_csv}")
    else:
        print("⚠️ No rows to save.")

    # 3) Resumen numérico
    summary = summarize_rows(out_rows)
    return summary


# ============= CLI simple =============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="resultados_emails.csv",
        help="CSV de entrada con resultados del RAG",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV de salida con columnas DUDU; por defecto input + '_dudu.csv'",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Opcional: número máximo de filas a evaluar (para pruebas).",
    )
    args = parser.parse_args()

    out = args.output or args.input.replace(".csv", "_dudu.csv")
    run_dudu_eval_for_file(args.input, out, max_rows=args.max_rows)
