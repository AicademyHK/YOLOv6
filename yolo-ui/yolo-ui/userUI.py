import gradio as gr
from onnxbackend import load_model, update_labels, process_image, load_labels
import os
from PIL import Image

def webcam_inference(image):
    if model is None:
        return image
    pil_image = Image.fromarray(image)
    result_text, processed_image = process_image(pil_image)
    return processed_image

def create_interface():
    labels = load_labels(os.path.join(os.getcwd(), "data/id_labels.ini"))
    labels_str = ",".join(labels)
    font_size = 1

    with gr.Blocks() as demo:
        with gr.Tab("Model & Inference"):
            with gr.Row():
                file_input = gr.File(label="Select ONNX Model", file_types=["onnx"], height=100)
                width_input = gr.Textbox(label="Input Width", placeholder="640", type="text", value="640")
                height_input = gr.Textbox(label="Input Height", placeholder="640", type="text", value="640")
                provider_dropdown = gr.Dropdown(label="Select Execution Provider", choices=["CPU", "CUDA", "TensorRT"])
                load_button = gr.Button("Load Model")

            with gr.Row():
                label_input = gr.Textbox(label="Enter Class Labels (comma-separated)", placeholder="person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light", value=labels_str)
                font_size_slider = gr.Slider(label="Label Font Size", minimum=0.5, maximum=3, value=1, step=0.1)
                update_button = gr.Button("Update Labels")

            with gr.Row():
                image_input = gr.Image(sources=["upload"], label="Upload Image")
                image_output = gr.Image(label="Processed Image")
                infer_button = gr.Button("Run Inference")

            # Set up interactions
            load_button.click(
                lambda file, width, height, provider: gr.update(value=load_model(file, width, height, provider)),
                inputs=[file_input, width_input, height_input, provider_dropdown],
                outputs=[]
            )
            update_button.click(
                lambda label_str, font_size: gr.update(value=update_labels(*label_str.split(',')), font_size=font_size),
                inputs=[label_input, font_size_slider],
                outputs=[]
            )
            infer_button.click(lambda img, font_size: process_image(img, font_size), inputs=[image_input, font_size_slider], outputs=image_output)
            

        with gr.Tab("Real-Time Detection"):
            with gr.Row():
                webcam_input = gr.Image(sources=["webcam"], label="Webcam Input")
                webcam_output = gr.Image(label="Webcam Output")
            webcam_input.change(lambda img, font_size: webcam_inference(img, font_size), inputs=[webcam_input, font_size_slider], outputs=webcam_output)

    return demo