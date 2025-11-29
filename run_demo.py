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
DEVICE = "cpu" #
print(f"⚙️ Usando dispositivo: {DEVICE}")
# =================================================

# Variables globales
global_batch = None
model = None

def process_next_batch(question: str):
    global global_batch, model
    
    if global_batch is None:
        return ([], "⚠️ Primero sube un PDF.", [], "", [], "", "Error: No PDF loaded", 0.0, "")
    
    # 1. Inyectar la pregunta en el batch
    global_batch["questions"] = [question]
    print(f"🤖 Preguntando: '{question}'...")

    try:
        # 2. Inferencia (Magia RAG)
        outputs, pred_answers, _, pred_conf, retrieval = model.inference(
            global_batch,
            return_retrieval=True,
            return_steps=True
        )
    except Exception as e:
        import traceback
        print("\n❌ ERROR DETALLADO:")
        traceback.print_exc()
        print(f"Mensaje corto: {e}\n")
        # =============================================
        return ([], f"Error crítico en inferencia: {str(e)}", [], "", [], "", "Error", 0.0, "")

    # 3. Extraer resultados para mostrar
    original_images = global_batch["images"][0]
    
    # Recuperación
    retrieved_patches = retrieval["patches"][0]
    retrieved_text_list = retrieval["text"][0]
    retrieved_chunks_str = "\n---\n".join([f"Chunk {i+1}:\n{txt}" for i, txt in enumerate(retrieved_text_list)])
    
    # Índices de páginas usadas
    # Manejo seguro si 'page_indices' no viene en el formato esperado
    page_indices = retrieval.get("page_indices", [[0]])
    if isinstance(page_indices, list) and len(page_indices) > 0:
        page_indices = page_indices[0]
    else:
        page_indices = [0]
        
    page_str = ", ".join([str(p+1) for p in page_indices])

    # Respuesta final
    predicted_answer = pred_answers[0]
    confidence = pred_conf[0] if pred_conf is not None else 0.0

    return (
        original_images,
        "Texto original cargado internamente.",
        [], "", 
        retrieved_patches,
        retrieved_chunks_str,
        predicted_answer,
        confidence,
        page_str
    )

def process_pdf_document(pdf_file):
    global global_batch
    if pdf_file is None: return
    
    gr.Info("⏳ Leyendo PDF con pdfminer... (puede tardar unos segundos)")
    
    try:
        record = load_pdf(pdf_file.name)
    except Exception as e:
        raise gr.Error(f"Error leyendo PDF: {e}. ¿Instalaste poppler-utils?")

    # Construimos el 'Contexto' uniendo palabras
    context_pages = []
    for page_tokens in record["ocr_tokens"]:
        context_pages.append(" ".join(page_tokens))

    # Armamos el Batch manual (Batch Size = 1)
    global_batch = {
        "question_id": [0],
        "questions": ["placeholder"],
        "answers": [[""]],
        "answer_page_idx": [[0]],
        "contexts": [context_pages],
        "images": [record["images"]],
        "words": [record["ocr_tokens"]],
        "boxes": [record["ocr_boxes"]],
        "num_pages": [len(record["images"])]
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
        gallery_orig = gr.Gallery(label="Páginas Originales")
        gallery_rag = gr.Gallery(label="Evidencia Visual (Chunks)")
        text_rag = gr.Textbox(label="Evidencia Texto", lines=5)

    upload_btn.upload(process_pdf_document, inputs=upload_btn, outputs=status_txt)
    
    ask_btn.click(process_next_batch, inputs=q_input, outputs=[
        gallery_orig, gr.Textbox(visible=False), gr.Gallery(visible=False), gr.Textbox(visible=False),
        gallery_rag, text_rag,
        ans_output, conf_output, pages_output
    ])

if __name__ == "__main__":
    # Configuración manual corregida
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
        # === PARÁMETROS AGREGADOS PARA CORREGIR EL KEYERROR ===
        "compute_stats": False,          # <--- FALTABA ESTE
        "compute_stats_examples": False, # <--- Y ESTE POR SEGURIDAD
        "n_stats_examples": 5,
        # ======================================================
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
    demo.launch(server_name="0.0.0.0", server_port=7860)