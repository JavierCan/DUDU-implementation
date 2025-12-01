import gradio as gr
import argparse
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Importaciones del proyecto
from src.build_utils import build_model
from src.utils import load_config
# Importamos la función original del proyecto
from src.process_pdf import load_pdf

# ================= CONFIGURACIÓN =================
DEVICE = "cpu"  # Puedes cambiar a 'cuda' si luego quieres GPU
print(f"⚙️ Usando dispositivo: {DEVICE}")
# =================================================

# Evitar que PIL reviente por PDFs grandes (solo quita el límite de seguridad)
Image.MAX_IMAGE_PIXELS = None

# Variables globales
global_batch = None
model = None


def resize_for_gallery(img, max_side=3000, max_pixels=40_000_000):
    """
    Redimensiona una imagen si es muy grande, para evitar errores al guardarla/mostrarla.
    - max_side: longitud máxima de ancho o alto.
    - max_pixels: máximo de píxeles totales (ancho*alto).
    Mantiene la proporción y usa LANCZOS para buena calidad.
    """
    if img is None:
        return None

    # Asegurar que sea un objeto PIL.Image
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))

    w, h = img.size
    if w <= 0 or h <= 0:
        return img

    # Si no es tan grande, no hacemos nada
    if (w * h) <= max_pixels and max(w, h) <= max_side:
        return img

    # Calculamos factor de escala usando ambas restricciones
    scale_side = max_side / max(w, h)
    scale_pixels = (max_pixels / float(w * h)) ** 0.5
    scale = min(scale_side, scale_pixels, 1.0)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    print(f"⚠️ Redimensionando imagen de {w}x{h} a {new_w}x{new_h} para la galería")

    try:
        # Pillow nuevo
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    except AttributeError:
        # Pillow viejo
        resized = img.resize((new_w, new_h), Image.LANCZOS)

    return resized


def process_next_batch(question: str):
    global global_batch, model

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

    # 1. Inyectar la pregunta en el batch
    global_batch["questions"] = [question]
    print(f"🤖 Preguntando: '{question}'...")

    try:
        # 2. Inferencia (Magia RAG)
        outputs, pred_answers, _, pred_conf, retrieval = model.inference(
            global_batch,
            return_retrieval=True,
            return_steps=True,
        )
    except Exception as e:
        import traceback

        print("\n❌ ERROR DETALLADO:")
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

    # 3. Extraer resultados para mostrar

    # Páginas originales
    # global_batch["images"] tiene forma [ [img_pagina_1, img_pagina_2, ...] ]
    original_images_raw = global_batch["images"][0]
    # Redimensionamos SOLO para la galería (para evitar imágenes gigantes)
    original_images = [resize_for_gallery(img) for img in original_images_raw]

    # Recuperación visual (patches/chunks)
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

    # Recuperación texto
    retrieved_text_list = retrieval.get("text", [[]])[0] if retrieval is not None else []
    retrieved_chunks_str = "\n---\n".join(
        [f"Chunk {i+1}:\n{txt}" for i, txt in enumerate(retrieved_text_list)]
    )

    # Índices de páginas usadas
    page_indices = retrieval.get("page_indices", [[0]]) if retrieval is not None else [[0]]
    if isinstance(page_indices, list) and len(page_indices) > 0:
        page_indices = page_indices[0]
    else:
        page_indices = [0]

    page_str = ", ".join([str(p + 1) for p in page_indices])

    # Respuesta final
    predicted_answer = pred_answers[0]
    confidence = pred_conf[0] if pred_conf is not None else 0.0

    return (
        original_images,           # gallery_orig
        "Texto original cargado internamente.",  # textbox oculto
        [],                        # gallery oculta (no la usamos)
        "",                        # textbox oculto
        retrieved_patches,         # gallery_rag
        retrieved_chunks_str,      # text_rag
        predicted_answer,          # ans_output
        confidence,                # conf_output
        page_str,                  # pages_output
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

    # Construimos el 'Contexto' uniendo palabras
    context_pages = []
    for page_tokens in record["ocr_tokens"]:
        context_pages.append(" ".join(page_tokens))

    # Redimensionamos las imágenes para interfaz (evita imágenes enormes en la Gallery)
    resized_images = [resize_for_gallery(img) for img in record["images"]]

    # Armamos el Batch manual (Batch Size = 1)
    global_batch = {
        "question_id": [0],
        "questions": ["placeholder"],
        "answers": [[""]],
        "answer_page_idx": [[0]],
        "contexts": [context_pages],
        "images": [resized_images],
        "words": [record["ocr_tokens"]],
        "boxes": [record["ocr_boxes"]],
        "num_pages": [len(resized_images)],
    }

    msg = f"✅ PDF Cargado: {len(record['images'])} páginas."
    print(msg)
    gr.Info(msg)
    return msg


# --- Interfaz Gráfica ---
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Demo Local RAG-DocVQA")

    with gr.Row():
        upload_btn = gr.UploadButton("📂 Subir PDF", file_types=[".pdf"])
        status_txt = gr.Textbox(label="Estado", interactive=False)

    with gr.Row():
        q_input = gr.Textbox(label="Tu Pregunta", placeholder="Ej: ¿Cuál es el monto total?")
        ask_btn = gr.Button("Enviar")

    with gr.Row():
        ans_output = gr.Textbox(label="Respuesta")
        conf_output = gr.Number(label="Confianza")
        pages_output = gr.Textbox(label="Páginas Usadas")

    with gr.Row():
        # Importante: forzamos PNG para evitar problemas con WebP
        gallery_orig = gr.Gallery(label="Páginas Originales", format="png")
        gallery_rag = gr.Gallery(label="Evidencia Visual (Chunks)", format="png")
        text_rag = gr.Textbox(label="Evidencia Texto", lines=5)

    # Cuando se sube el PDF
    upload_btn.upload(process_pdf_document, inputs=upload_btn, outputs=status_txt)

    # Cuando se hace la pregunta
    ask_btn.click(
        process_next_batch,
        inputs=q_input,
        outputs=[
            gallery_orig,
            gr.Textbox(visible=False),   # placeholder
            gr.Gallery(visible=False),   # placeholder
            gr.Textbox(visible=False),   # placeholder
            gallery_rag,
            text_rag,
            ans_output,
            conf_output,
            pages_output,
        ],
    )


if __name__ == "__main__":
    # Configuración manual
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
        # RUTAS Y MODELOS CLAVE
        "model_weights": "rubentito/vt5-base-spdocvqa",
        "embed_weights": "BAAI/bge-m3",
        "reranker_weights": "BAAI/bge-reranker-v2-m3",
        # Parámetros extra que te habías puesto para el KeyError
        "compute_stats": False,
        "compute_stats_examples": False,
        "n_stats_examples": 5,
        "layout_model": "DIT",
        "layout_model_weights": "microsoft/dit-base-finetuned-rvlcdip",
        "use_layout_labels": "Default",
        # Rutas dummy
        "imdb_dir": "./data",
        "images_dir": "./data",
    }

    args = argparse.Namespace(**args_dict)
    print("⏳ Cargando configuración y descargando modelos...")
    config = load_config(args)

    # Aseguramos que la config tenga layout_model por si load_config lo borra
    if "layout_model" not in config:
        config["layout_model"] = "DIT"

    model = build_model(config)
    model.to(DEVICE)
    model.eval()

    print("✅ ¡Listo! Abriendo interfaz...")
    demo.launch(server_name="0.0.0.0", server_port=7861)
