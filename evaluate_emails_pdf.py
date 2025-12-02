import csv

from src.build_utils import build_model
from src.utils import load_config
from src.process_pdf import load_pdf

from metrics_docvqa import (
    anls_single,
    retrieval_precision_at_k,
    chunk_score_at_k,
)

# ===================== FIXED PATHS =====================
DEVICE = "cuda"  # or "cuda" if you have a GPU

PDF_PATH = "Dataset sintético - Hilos de correo sobre retroalimentación de producto.pdf"
GT_CSV = "ground_truth_emails.csv"
OUT_CSV = "resultados_emails.csv"
# =======================================================


# -----------------------------
# 1. Load PDF and build batch
# -----------------------------
def build_batch_from_pdf(pdf_path: str):
    print(f"📄 Loading PDF: {pdf_path}")
    record = load_pdf(pdf_path)

    context_pages = []
    for page_tokens in record["ocr_tokens"]:
        context_pages.append(" ".join(page_tokens))

    batch = {
        "question_id": [0],
        "questions": ["placeholder"],
        "answers": [[""]],
        "answer_page_idx": [[0]],
        "contexts": [context_pages],
        "images": [record["images"]],
        "words": [record["ocr_tokens"]],
        "boxes": [record["ocr_boxes"]],
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

        pred_answer = pred_answers[0]
        conf = float(pred_conf[0]) if pred_conf is not None else 0.0

        # Retrieval
        retrieved_text_list = retrieval["text"][0]
        page_indices = retrieval.get("page_indices", [[0]])
        if isinstance(page_indices, list) and len(page_indices) > 0:
            page_indices = page_indices[0]
        else:
            page_indices = [0]

        k = min(k_chunks, len(retrieved_text_list))
        topk_chunks_text = "\n\n---\n\n".join(retrieved_text_list[:k])

        # METRICS
        anls_val = anls_single(pred_answer, gt_answer)
        ret_prec = retrieval_precision_at_k(gt_page, page_indices, k)
        chunk_sc = chunk_score_at_k(retrieved_text_list, gt_answer, k)

        print(f"   GT:   {gt_answer}")
        print(f"   Pred: {pred_answer}")
        print(
            f"   ANLS: {anls_val:.3f} | "
            f"Ret@k: {ret_prec:.3f} | "
            f"ChunkScore@k: {chunk_sc:.3f}"
        )

        all_results.append(
            {
                "question": q,
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
                "gt_page_idx": gt_page,
                "pages_pred": ";".join(str(p) for p in page_indices[:k]),
                "anls": anls_val,
                "retrieval_precision@k": ret_prec,
                "chunk_score@k": chunk_sc,
                "confidence": conf,
                "retrieved_chunks": topk_chunks_text,  # <- for DUDU / LLM-judge
            }
        )

    # Global summary
    if all_results:
        avg_anls = sum(r["anls"] for r in all_results) / len(all_results)
        avg_ret = sum(r["retrieval_precision@k"] for r in all_results) / len(all_results)
        avg_chunk = sum(r["chunk_score@k"] for r in all_results) / len(all_results)
        print("\n📊 GLOBAL SUMMARY")
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
    # Model configuration (same as your Gradio demo)
    args_dict = {
        "model": "RAGVT5",
        "dataset": "MP-DocVQA",
        "embed_model": "BGE",
        "page_retrieval": "Concat",
        "add_sep_token": False,
        "layout_batch_size": 2,
        "chunk_num": 10,
        "chunk_size": 60,
        "chunk_size_tol": 0.2,
        "overlap": 10,
        "include_surroundings": 0,
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

    import argparse as _ap
    cfg = load_config(_ap.Namespace(**args_dict))
    if "layout_model" not in cfg:
        cfg["layout_model"] = "DIT"

    print("⏳ Loading model...")
    model = build_model(cfg)
    model.to(DEVICE)
    model.eval()
    print("✅ Model ready.")

    batch = build_batch_from_pdf(PDF_PATH)
    gt_examples = load_gt_from_csv(GT_CSV)

    results = eval_on_pdf(model, batch, gt_examples, k_chunks=10)
    save_results(results, OUT_CSV)


if __name__ == "__main__":
    main()
