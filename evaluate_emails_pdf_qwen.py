import csv
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.build_utils import build_model
from src.utils import load_config
from src.process_pdf import load_pdf

from metrics_docvqa import (
    anls_single,
    retrieval_precision_at_k,
    chunk_score_at_k,
)

# ===================== FIXED PATHS =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PDF_PATH = "Dataset sintético - Hilos de correo sobre retroalimentación de producto.pdf"
GT_CSV = "ground_truth_emails.csv"
OUT_CSV = "resultados_emails_qwen.csv"   # nuevo archivo para este experimento
# =======================================================

# ================== QWEN CONFIG ========================
QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # el modelo grande que estás usando
qwen_tokenizer = None
qwen_model = None


def init_qwen():
    """Carga Qwen solo una vez en GPU/CPU."""
    global qwen_tokenizer, qwen_model
    if qwen_model is not None:
        return

    print(f"⏳ Cargando Qwen: {QWEN_MODEL_NAME} ...")
    torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)

    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    if DEVICE != "cuda":
        qwen_model.to(DEVICE)

    qwen_model.eval()
    print("✅ Qwen cargado.")


def answer_with_qwen(question: str, context: str) -> str:
    global qwen_tokenizer, qwen_model
    if qwen_model is None:
        init_qwen()

    messages = [
        {
            "role": "system",
            "content": (
                "Eres un asistente de QA sobre documentos. "
                "Responde SOLO usando la información del contexto. "
                "Responde con una frase corta, sin explicaciones, "
                "en el mismo idioma de la pregunta. "
                "No digas 'assistant'. "
                "Si la respuesta no está claramente en el contexto, responde "
                "EXACTAMENTE: NO_ENCONTRADO."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Contexto:\n{context}\n\n"
                f"Pregunta: {question}\n\n"
                "Respuesta:"
            ),
        },
    ]

    prompt = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = qwen_tokenizer(prompt, return_tensors="pt").to(qwen_model.device)

    with torch.no_grad():
        output_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4,
            do_sample=False,          # determinista
            temperature=0.0,          # quita los warnings de sampling
            top_p=1.0,
            top_k=0,
            eos_token_id=qwen_tokenizer.eos_token_id,
        )

    full = qwen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "Respuesta:" in full:
        ans = full.split("Respuesta:")[-1].strip()
    else:
        ans = full.strip()

    ans = clean_model_answer(ans)
    return ans

# =======================================================
def clean_model_answer(text: str) -> str:
    """Limpia el prefijo 'assistant', saltos de línea raros, etc."""
    if not text:
        return ""

    text = text.strip()

    # quitar cabeceras típicas de chat
    lower = text.lower()
    for prefix in ["assistant", "assistant:", "assistant\n", "assistant:\n"]:
        if lower.startswith(prefix):
            # recortar longitud real del prefijo original
            text = text[len(prefix):].lstrip(" :\n\t")
            break

    # quitar comillas sueltas y espacios extra
    text = text.strip().strip('"').strip()
    return text


# -----------------------------
# 1. Load PDF and build batch
# -----------------------------
def build_batch_from_pdf(pdf_path: str):
    print(f"📄 Loading PDF: {pdf_path}")
    record = load_pdf(pdf_path)

    num_pages = len(record["images"])
    ocr_tokens = record.get("ocr_tokens", None)

    if ocr_tokens and len(ocr_tokens) == num_pages:
        context_pages = []
        for page_tokens in ocr_tokens:
            if page_tokens is None:
                context_pages.append("")
            else:
                context_pages.append(" ".join(str(t) for t in page_tokens))
    else:
        print("⚠️ No hay ocr_tokens válidos; usando contexto vacío por página.")
        context_pages = ["" for _ in range(num_pages)]

    batch = {
        "question_id": [0],
        "questions": ["placeholder"],
        "answers": [[""]],
        "answer_page_idx": [[0]],
        "contexts": [context_pages],
        "images": [record["images"]],
        "words": [ocr_tokens if ocr_tokens is not None else []],
        "boxes": [record.get("ocr_boxes", [])],
        "num_pages": [len(record["images"])],
    }

    print(f"✅ PDF loaded with {len(record['images'])} pages.")
    return batch


# -----------------------------
# 2. Load ground truth CSV
# -----------------------------
def load_gt_from_csv(csv_path: str):
    gt_examples = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"]
            a = row["answer"]
            page_num = int(row["page"])
            gt_examples.append(
                {
                    "question": q,
                    "answer": a,
                    "page_idx": page_num - 1,  # 0-based index
                }
            )
    print(f"✅ Ground truth loaded: {len(gt_examples)} questions.")
    return gt_examples


# -----------------------------
# 3. Evaluation over one PDF
# -----------------------------
def eval_on_pdf(model, batch, gt_examples, k_chunks=10):
    """
    Igual que antes, pero:
    - usamos RAGVT5 para retrieval,
    - VT5 genera una respuesta baseline,
    - Qwen genera la respuesta final sobre los top-k chunks.
    """
    all_results = []

    for idx, ex in enumerate(gt_examples):
        q = ex["question"]
        gt_answer = ex["answer"]
        gt_page = ex["page_idx"]

        print(f"\n➡️ [{idx+1}/{len(gt_examples)}] Question: {q}")

        # Inject the question into the batch
        batch["questions"] = [q]

        outputs, pred_answers, _, pred_conf, retrieval = model.inference(
            batch, return_retrieval=True, return_steps=True
        )

        vt5_answer = pred_answers[0]
        conf = float(pred_conf[0]) if pred_conf is not None else 0.0

        # Retrieval
        retrieved_text_list = retrieval["text"][0]
        page_indices = retrieval.get("page_indices", [[0]])
        if isinstance(page_indices, list) and len(page_indices) > 0:
            page_indices = page_indices[0]
        else:
            page_indices = [0]

        k = min(k_chunks, len(retrieved_text_list))
        # contexto textual que le pasamos a Qwen (solo top-k para no pasarnos de tokens)
        topk_chunks_text = "\n\n---\n\n".join(retrieved_text_list[:k])

        # ===== Respuesta con Qwen sobre el contexto =====
        qwen_answer = answer_with_qwen(q, topk_chunks_text)
        print(f"   VT5 pred:  {vt5_answer!r}")
        print(f"   Qwen pred: {qwen_answer!r}")

        normalized = qwen_answer.strip().upper()

        if (normalized
            and "NO_ENCONTRADO" not in normalized
            and "EXACTAMENTE: NO_ENCONTRADO" not in normalized):
            final_answer = qwen_answer
        else:
            final_answer = vt5_answer
        print("⚠️ Qwen devolvió NO_ENCONTRADO o vacío; usando VT5 como respaldo.")


        # METRICS (para Qwen+fallback y para VT5 baseline)
        anls_val = anls_single(final_answer, gt_answer)
        vt5_anls_val = anls_single(vt5_answer, gt_answer)

        ret_prec = retrieval_precision_at_k(gt_page, page_indices, k)
        chunk_sc = chunk_score_at_k(retrieved_text_list, gt_answer, k)

        print(f"   GT:        {gt_answer}")
        print(f"   FINAL:     {final_answer}")
        print(
            f"   ANLS(final): {anls_val:.3f} | ANLS(VT5): {vt5_anls_val:.3f} | "
            f"Ret@k: {ret_prec:.3f} | ChunkScore@k: {chunk_sc:.3f}"
        )

        all_results.append(
            {
                "question": q,
                "gt_answer": gt_answer,
                "pred_answer": final_answer,      # <-- lo que usará DUDU
                "vt5_answer": vt5_answer,         # <-- extra para análisis
                "gt_page_idx": gt_page,
                "pages_pred": ";".join(str(p) for p in page_indices[:k]),
                "anls": anls_val,                 # ANLS de la respuesta final
                "vt5_anls": vt5_anls_val,         # ANLS original de VT5
                "retrieval_precision@k": ret_prec,
                "chunk_score@k": chunk_sc,
                "confidence": conf,
                "retrieved_chunks": topk_chunks_text,  # para DUDU / LLM-judge
            }
        )

    # Global summary
    if all_results:
        avg_anls = sum(r["anls"] for r in all_results) / len(all_results)
        avg_ret = sum(r["retrieval_precision@k"] for r in all_results) / len(all_results)
        avg_chunk = sum(r["chunk_score@k"] for r in all_results) / len(all_results)
        print("\n📊 GLOBAL SUMMARY (Qwen+fallback)")
        print(f"- Average ANLS: {avg_anls:.3f}")
        print(f"- Average Retrieval Precision@k: {avg_ret:.3f}")
        print(f"- Average Chunk Score@k: {avg_chunk:.3f}")

    return all_results


# -----------------------------
# 4. Save results to CSV
# -----------------------------
def save_results(results, out_csv: str):
    if not results:
        print("⚠️ No results to save.")
        return

    fieldnames = list(results[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"💾 Results saved to: {out_csv}")


# -----------------------------
# 5. main without argparse
# -----------------------------
def main():
    # Model configuration (más “turbo”, como en run_demo_qwen)
    args_dict = {
        "model": "RAGVT5",
        "dataset": "MP-DocVQA",
        "embed_model": "BGE",
        "page_retrieval": "Concat",
        "add_sep_token": False,
        "layout_batch_size": 4,
        "chunk_num": 20,
        "chunk_size": 60,
        "chunk_size_tol": 0.2,
        "overlap": 10,
        "include_surroundings": 1,
        "device": DEVICE,
        "model_weights": "rubentito/vt5-base-spdocvqa",
        "embed_weights": "BAAI/bge-m3",
        "reranker_weights": "BAAI/bge-reranker-v2-m3",
        "compute_stats": False,
        "compute_stats_examples": False,
        "n_stats_examples": 5,
        "layout_model": "DIT",
        "layout_model_weights": "microsoft/dit-base-finetuned-rvlcdip",
        "use_layout_labels": "Default",
        "imdb_dir": "./data",
        "images_dir": "./data",
    }

    cfg = load_config(argparse.Namespace(**args_dict))
    if "layout_model" not in cfg:
        cfg["layout_model"] = "DIT"

    print("⏳ Loading RAGVT5 model...")
    model = build_model(cfg)
    model.to(DEVICE)
    model.eval()
    print("✅ RAGVT5 ready.")

    batch = build_batch_from_pdf(PDF_PATH)
    gt_examples = load_gt_from_csv(GT_CSV)

    results = eval_on_pdf(model, batch, gt_examples, k_chunks=10)
    save_results(results, OUT_CSV)


if __name__ == "__main__":
    main()
