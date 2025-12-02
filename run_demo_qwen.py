import gradio as gr
import argparse
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Proyecto RAG-DocVQA
from src.build_utils import build_model
from src.utils import load_config
from src.process_pdf import load_pdf

# Qwen como generador
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= CONFIGURACIÓN =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Usando dispositivo principal: {DEVICE}")
Image.MAX_IMAGE_PIXELS = None
# =================================================

# --- Modelo RAGVT5 (retriever + layout + embeddings) ---
global_batch = None
rag_model = None

# --- Modelo Qwen (generador) ---
QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
qwen_tokenizer = None
qwen_model = None


def init_qwen():
    """
    Carga Qwen solo una vez.
    """
    global qwen_tokenizer, qwen_model

    if qwen_model is not None:
        return

    print(f"⏳ Cargando Qwen: {QWEN_MODEL_NAME} ...")
    torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)

    # device_map="auto" reparte en GPU si está disponible
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
    """
    Usa Qwen para responder usando SOLO el contexto recuperado por RAG.
    """
    global qwen_tokenizer, qwen_model
    if qwen_model is None:
        init_qwen()

    messages = [
        {
            "role": "system",
            "content": (
                "Eres un asistente de QA sobre documentos. "
                "Responde SOLO usando la información del contexto. "
                "Responde con una frase corta en el mismo idioma de la pregunta. "
                "Si la respuesta no está claramente en el contexto, responde EXACTAMENTE: NO_ENCONTRADO."
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
            eos_token_id=qwen_tokenizer.eos_token_id,
        )

    full = qwen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Nos quedamos con lo que viene después de "Respuesta:"
    if "Respuesta:" in full:
        ans = full.split("Respuesta:")[-1].strip()
    else:
        ans = full.strip()

    ans = ans.replace("Respuesta:", "").strip()
    return ans


def resize_for_gallery(img, max_side=3000, max_pixels=40_000_000):
    if img is None:
        return None

    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))

    w, h = img.size
    if w <= 0 or h <= 0:
        return img

    if (w * h) <= max_pixels and max(w, h) <= max_side:
        return img

    scale_side = max_side / max(w, h)
    scale_pixels = (max_pixels / float(w * h)) ** 0.5
    scale = min(scale_side, scale_pixels, 1.0)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    print(f"⚠️ Redimensionando imagen de {w}x{h} a {new_w}x{new_h} para la galería")

    resample_filter = getattr(
        Image, "LANCZOS",
        getattr(Image, "ANTIALIAS", Image.BICUBIC)
    )

    resized = img.resize((new_w, new_h), resample_filter)
    return resized


def process_next_batch(question: str):
    global global_batch, rag_model

    if global_batch is None:
        return (
            [],
            "⚠️ Primero sube un PDF.",
            [],
            "",
            [],
            "",
            "Error: No PDF loaded",
            0.0,
            "",
        )

    global_batch["questions"] = [question]
    print(f"🤖 Preguntando (RAG + Qwen): '{question}'...")

    try:
        # 1) RAGVT5 hace retrieval y también genera una respuesta (que usaremos solo como backup)
        outputs, pred_answers, _, pred_conf, retrieval = rag_model.inference(
            global_batch,
            return_retrieval=True,
            return_steps=True,
        )
    except Exception as e:
        import traceback

        print("\n❌ ERROR DETALLADO RAGVT5:")
        traceback.print_exc()
        print(f"Mensaje corto: {e}\n")
        return (
            [],
            f"Error crítico en inferencia: {str(e)}",
            [],
            "",
            [],
            "",
            "Error",
            0.0,
            "",
        )

    # --- Imágenes originales ---
    original_images_raw = global_batch["images"][0]
    original_images = [resize_for_gallery(img) for img in original_images_raw]

    # --- Patches visuales (chunks) ---
    retrieved_patches_raw = []
    if retrieval is not None and "patches" in retrieval and retrieval["patches"]:
        try:
            retrieved_patches_raw = retrieval["patches"][0]
        except Exception as e:
            print("⚠️ No se pudieron leer los patches de retrieval:", e)
            retrieved_patches_raw = []

    retrieved_patches = []
    for p in retrieved_patches_raw:
        try:
            retrieved_patches.append(resize_for_gallery(p))
        except Exception as e:
            print("⚠️ No se pudo redimensionar un patch:", e)

    # --- Texto recuperado ---
    retrieved_text_list = retrieval.get("text", [[]])[0] if retrieval is not None else []
    retrieved_chunks_str = "\n---\n".join(
        [f"Chunk {i+1}:\n{txt}" for i, txt in enumerate(retrieved_text_list)]
    )

    # --- Índices de página ---
    page_indices = retrieval.get("page_indices", [[0]]) if retrieval is not None else [[0]]
    if isinstance(page_indices, list) and len(page_indices) > 0:
        page_indices = page_indices[0]
    else:
        page_indices = [0]
    page_str = ", ".join([str(p + 1) for p in page_indices])

    # --- Respuestas VT5 vs Qwen ---
    predicted_answer_vt5 = pred_answers[0]
    confidence = pred_conf[0] if pred_conf is not None else 0.0

    answer_qwen = answer_with_qwen(question, retrieved_chunks_str)
    print(f"🧠 VT5 dijo: {predicted_answer_vt5!r}")
    print(f"🪄 Qwen dijo: {answer_qwen!r}")

    if answer_qwen and answer_qwen != "NO_ENCONTRADO":
        predicted_answer = answer_qwen
    else:
        predicted_answer = predicted_answer_vt5
        print("⚠️ Qwen devolvió NO_ENCONTRADO o vacío; usando VT5 como respaldo.")

    return (
        original_images,
        "Texto original cargado internamente.",
        [],
        "",
        retrieved_patches,
        retrieved_chunks_str,
        predicted_answer,
        confidence,
        page_str,
    )


def process_pdf_document(pdf_file):
    global global_batch
    if pdf_file is None:
        return

    gr.Info("⏳ Leyendo PDF con pdfminer... (puede tardar unos segundos)")

    try:
        record = load_pdf(pdf_file.name)
    except Exception as e:
        raise gr.Error(f"Error leyendo PDF: {e}. ¿Instalaste poppler-utils?")

    num_pages = len(record["images"])

    # Contextos por página
    context_pages = []
    ocr_tokens = record.get("ocr_tokens", None)

    if ocr_tokens and len(ocr_tokens) == num_pages:
        for page_tokens in ocr_tokens:
            if page_tokens is None:
                context_pages.append("")
            else:
                context_pages.append(" ".join(str(t) for t in page_tokens))
    else:
        print("⚠️ No hay ocr_tokens válidos; usando contexto vacío por página.")
        context_pages = ["" for _ in range(num_pages)]

    resized_images = [resize_for_gallery(img) for img in record["images"]]

    global_batch = {
        "question_id": [0],
        "questions": ["placeholder"],
        "answers": [[""]],
        "answer_page_idx": [[0]],
        "contexts": [context_pages],
        "images": [resized_images],
        "words": [ocr_tokens if ocr_tokens is not None else []],
        "boxes": [record.get("ocr_boxes", [])],
        "num_pages": [len(resized_images)],
    }

    msg = f"✅ PDF Cargado: {len(record['images'])} páginas."
    print(msg)
    gr.Info(msg)
    return msg


# ----------------- UI GRADIO -----------------
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Demo Local RAG-DocVQA (RAGVT5 + Qwen)")

    with gr.Row():
        upload_btn = gr.UploadButton("📂 Subir PDF", file_types=[".pdf"])
        status_txt = gr.Textbox(label="Estado", interactive=False)

    with gr.Row():
        q_input = gr.Textbox(label="Tu Pregunta", placeholder="Ej: ¿Cuál es el monto total?")
        ask_btn = gr.Button("Enviar")

    with gr.Row():
        ans_output = gr.Textbox(label="Respuesta (Qwen + RAG)")
        conf_output = gr.Number(label="Confianza (VT5)")
        pages_output = gr.Textbox(label="Páginas Usadas")

    with gr.Row():
        gallery_orig = gr.Gallery(label="Páginas Originales", format="png")
        gallery_rag = gr.Gallery(label="Evidencia Visual (Chunks)", format="png")
        text_rag = gr.Textbox(label="Evidencia Texto", lines=5)

    upload_btn.upload(process_pdf_document, inputs=upload_btn, outputs=status_txt)

    ask_btn.click(
        process_next_batch,
        inputs=q_input,
        outputs=[
            gallery_orig,
            gr.Textbox(visible=False),
            gr.Gallery(visible=False),
            gr.Textbox(visible=False),
            gallery_rag,
            text_rag,
            ans_output,
            conf_output,
            pages_output,
        ],
    )


if __name__ == "__main__":
    # ---- Config RAGVT5 igual que antes, pero con más contexto ----
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
        "embed_device": DEVICE,
        "reranker_device": DEVICE,
        "layout_device": DEVICE,

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

    args = argparse.Namespace(**args_dict)
    print("⏳ Cargando configuración RAGVT5...")
    config = load_config(args)
    if "layout_model" not in config:
        config["layout_model"] = "DIT"

    rag_model = build_model(config)
    rag_model.to(DEVICE)
    rag_model.eval()
    print("✅ RAGVT5 listo.")

    # Cargamos Qwen una vez al inicio
    init_qwen()

    print("✅ ¡Listo! Abriendo interfaz...")
    demo.launch(server_name="0.0.0.0", server_port=7862, share=True)
