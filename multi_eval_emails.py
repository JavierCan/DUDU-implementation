import csv
import argparse
import math
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.build_utils import build_model
from src.utils import load_config
from src.process_pdf import load_pdf

# 👇 NUEVO: importamos el DUDU-eval
from dudu_eval import run_dudu_eval_for_file

# ============================================================
#  MÉTRICAS (las mismas que ya usas para tu dataset)
# ============================================================

def levenshtein_distance(s: str, t: str) -> int:
    s = s or ""
    t = t or ""
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def normalized_lev_sim(pred: str, gt: str) -> float:
    pred = (pred or "").strip().lower()
    gt = (gt or "").strip().lower()

    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0

    d = levenshtein_distance(pred, gt)
    return 1.0 - d / max(len(pred), len(gt))


def anls_single(pred: str, gt: str, tau: float = 0.5) -> float:
    sim = normalized_lev_sim(pred, gt)
    return sim if sim >= tau else 0.0


def retrieval_precision_at_k(gt_page: int, pred_pages: list[int], k: int) -> float:
    if not pred_pages:
        return 0.0
    k = min(k, len(pred_pages))
    for j in range(k):
        if pred_pages[j] == gt_page:
            return 1.0
    return 0.0


def substring_max_sim(chunk: str, answer: str) -> float:
    chunk = (chunk or "").lower()
    answer = (answer or "").lower()
    L = len(answer)
    if L == 0 or len(chunk) == 0 or len(chunk) < L:
        return 0.0

    best = 0.0
    for i in range(0, len(chunk) - L + 1):
        sub = chunk[i : i + L]
        sim = normalized_lev_sim(sub, answer)
        if sim > best:
            best = sim
    return best


def chunk_score_at_k(retrieved_chunks: list[str], answer: str, k: int) -> float:
    if not retrieved_chunks:
        return 0.0
    k = min(k, len(retrieved_chunks))
    sims = [substring_max_sim(retrieved_chunks[j], answer) for j in range(k)]
    max_sim = max(sims) if sims else 0.0
    return math.log2(1.0 + max_sim)

# ============================================================
#  CONFIG GLOBAL: DATASETS + VARIANTES DE MODELO
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = [
    {
        "name": "emails_es",
        "pdf_path": "Dataset sintético - Hilos de correo sobre retroalimentación de producto.pdf",
        "gt_csv": "ground_truth_emails.csv",
    },
    {
        "name": "emails_en",
        "pdf_path": "Synthetic Dataset - Product Feedback Email Threads.pdf",
        "gt_csv": "ground_truth_emails_en.csv",
    },
]

EXPERIMENTS = [
    {
        "tag": "rag_vt5",        # RAG-VT5 (solo VT5)
        "use_qwen": False,
    },
    {
        "tag": "rag_vt5_qwen",   # RAG-VT5 + Qwen2.5-7B-Instruct
        "use_qwen": True,
    },
]

QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
qwen_tokenizer = None
qwen_model = None

# ============================================================
#  QWEN HELPERS
# ============================================================

def init_qwen():
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


def clean_model_answer(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    lower = text.lower()
    for prefix in ["assistant", "assistant:", "assistant\n", "assistant:\n"]:
        if lower.startswith(prefix):
            text = text[len(prefix):].lstrip(" :\n\t")
            break
    text = text.strip().strip('"').strip()
    return text


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
            do_sample=False,
            temperature=0.0,
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

# ============================================================
#  HELPERS PARA TU DATASET (PDF + CSV GT)
# ============================================================

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
                    "page_idx": page_num - 1,  # 0-based
                }
            )
    print(f"✅ Ground truth loaded: {len(gt_examples)} questions.")
    return gt_examples

# ============================================================
#  EVALUACIÓN SOBRE 1 PDF + 1 MODELO (variant)
# ============================================================

def eval_on_pdf(model, batch, gt_examples, k_chunks=10, use_qwen=False):
    all_results = []

    for idx, ex in enumerate(gt_examples):
        q = ex["question"]
        gt_answer = ex["answer"]
        gt_page = ex["page_idx"]

        print(f"\n➡️ [{idx+1}/{len(gt_examples)}] Question: {q}")

        batch["questions"] = [q]

        outputs, pred_answers, _, pred_conf, retrieval = model.inference(
            batch, return_retrieval=True, return_steps=True
        )

        vt5_answer = pred_answers[0]
        conf = float(pred_conf[0]) if pred_conf is not None else 0.0

        retrieved_text_list = retrieval["text"][0]
        page_indices = retrieval.get("page_indices", [[0]])
        if isinstance(page_indices, list) and len(page_indices) > 0:
            page_indices = page_indices[0]
        else:
            page_indices = [0]

        k = min(k_chunks, len(retrieved_text_list))
        topk_chunks_text = "\n\n---\n\n".join(retrieved_text_list[:k])

        if use_qwen:
            qwen_answer = answer_with_qwen(q, topk_chunks_text)
            print(f"   VT5 pred:  {vt5_answer!r}")
            print(f"   Qwen pred: {qwen_answer!r}")

            normalized = qwen_answer.strip().upper()
            if normalized and "NO_ENCONTRADO" not in normalized:
                final_answer = qwen_answer
            else:
                final_answer = vt5_answer
                print("⚠️ Qwen devolvió NO_ENCONTRADO o vacío; usando VT5 como respaldo.")
        else:
            qwen_answer = ""
            final_answer = vt5_answer
            print(f"   VT5 pred: {vt5_answer!r}")

        anls_val = anls_single(final_answer, gt_answer)
        vt5_anls_val = anls_single(vt5_answer, gt_answer)
        ret_prec = retrieval_precision_at_k(gt_page, page_indices, k)
        chunk_sc = chunk_score_at_k(retrieved_text_list, gt_answer, k)

        print(f"   GT:        {gt_answer}")
        print(
            f"   ANLS(final): {anls_val:.3f} | ANLS(VT5): {vt5_anls_val:.3f} | "
            f"Ret@k: {ret_prec:.3f} | ChunkScore@k: {chunk_sc:.3f}"
        )

        all_results.append(
            {
                "question": q,
                "gt_answer": gt_answer,
                "pred_answer": final_answer,
                "vt5_answer": vt5_answer,
                "qwen_answer": qwen_answer,
                "gt_page_idx": gt_page,
                "pages_pred": ";".join(str(p) for p in page_indices[:k]),
                "anls": anls_val,
                "vt5_anls": vt5_anls_val,
                "retrieval_precision@k": ret_prec,
                "chunk_score@k": chunk_sc,
                "confidence": conf,
                "retrieved_chunks": topk_chunks_text,
            }
        )

    if all_results:
        avg_anls = sum(r["anls"] for r in all_results) / len(all_results)
        avg_ret = sum(r["retrieval_precision@k"] for r in all_results) / len(all_results)
        avg_chunk = sum(r["chunk_score@k"] for r in all_results) / len(all_results)
        print("\n📊 GLOBAL SUMMARY")
        print(f"- Average ANLS: {avg_anls:.3f}")
        print(f"- Average Retrieval Precision@k: {avg_ret:.3f}")
        print(f"- Average Chunk Score@k: {avg_chunk:.3f}")

    return all_results


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

# ============================================================
#  MAIN: MULTI-EVAL + DUDU
# ============================================================

def main():
    # Configuración de RAG-VT5 compartida para todos los experimentos
    args_dict = {
        "model": "RAGVT5",
        "dataset": "MP-DocVQA",  # dummy
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

    print("⏳ Loading RAGVT5 model (compartido para todos los runs)...")
    model = build_model(cfg)
    model.to(DEVICE)
    model.eval()
    print("✅ RAGVT5 ready.")

    summary_rows = []

    for ds in DATASETS:
        pdf_path = ds["pdf_path"]
        gt_csv = ds["gt_csv"]
        ds_name = ds["name"]

        batch = build_batch_from_pdf(pdf_path)
        gt_examples = load_gt_from_csv(gt_csv)

        for exp in EXPERIMENTS:
            tag = exp["tag"]
            use_qwen = exp["use_qwen"]

            print("\n" + "=" * 80)
            print(f"🏁 DATASET: {ds_name} | EXPERIMENTO: {tag}")
            print("=" * 80)

            # 1) Eval clásica (ANLS, Ret@k, ChunkScore)
            results = eval_on_pdf(
                model,
                batch,
                gt_examples,
                k_chunks=10,
                use_qwen=use_qwen,
            )

            out_csv = f"results_{ds_name}_{tag}.csv"
            save_results(results, out_csv)

            avg_anls = sum(r["anls"] for r in results) / len(results)
            avg_ret = sum(r["retrieval_precision@k"] for r in results) / len(results)
            avg_chunk = sum(r["chunk_score@k"] for r in results) / len(results)

            # 2) DUDU-eval con Gemini sobre ese CSV
            dudu_csv = out_csv.replace(".csv", "_dudu.csv")
            print(f"\n🧠 Running DUDU eval for {out_csv} ...")
            dudu_summary = run_dudu_eval_for_file(out_csv, dudu_csv)

            # 3) Guardamos todo para la tabla final
            summary_rows.append(
                {
                    "dataset": ds_name,
                    "exp": tag,
                    "avg_anls": avg_anls,
                    "avg_ret": avg_ret,
                    "avg_chunk": avg_chunk,
                    "avg_E": dudu_summary["avg_E"],
                    "avg_C": dudu_summary["avg_C"],
                    "avg_H": dudu_summary["avg_H"],
                    "avg_DUDU": dudu_summary["avg_DUDU"],
                    "out_csv": out_csv,
                    "dudu_csv": dudu_csv,
                }
            )

    # ================= TABLA FINAL BONITA =================
    df = pd.DataFrame(summary_rows)
    df = df[
        [
            "dataset",
            "exp",
            "avg_anls",
            "avg_ret",
            "avg_chunk",
            "avg_E",
            "avg_C",
            "avg_H",
            "avg_DUDU",
            "out_csv",
            "dudu_csv",
        ]
    ]

    # Guardar como CSV
    df.to_csv("multi_eval_summary.csv", index=False)

    print("\n\n================= MULTI-EVAL SUMMARY =================")
    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )
    print("\n💾 Tabla completa guardada en: multi_eval_summary.csv")


if __name__ == "__main__":
    main()
