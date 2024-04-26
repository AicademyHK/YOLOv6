import onnxruntime as ort
import numpy as np
from PIL import Image
import configparser
import os
import cv2 as cv

# Global variables
model = None
labels = []
input_width = 640
input_height = 640

# Helper function to format the screenshot to the correct size and format for the model
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv.resize(im, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = np.asarray(cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=color))  # add border
    #return im, r, (dw, dh)
    return im

CLASS_COLORS = [
    (255, 0, 0),    # Class 0: Blue
    (0, 255, 0),    # Class 1: Green
    (0, 0, 255),    # Class 2: Red
    (255, 255, 0),  # Class 3: Cyan
    (255, 0, 255),  # Class 4: Magenta
    (0, 255, 255),  # Class 5: Yellow
    (128, 0, 0),    # Class 6: Dark Blue
    (0, 128, 0),    # Class 7: Dark Green
    (0, 0, 128),    # Class 8: Dark Red
    (128, 128, 0)   # Class 9: Dark Cyan
]

def get_class_color(class_id):
    return CLASS_COLORS[class_id % 10]

def draw_boxes(img, boxes, class_ids, scores, font_size):
    """
    Draw bounding boxes and labels on the image.
    """
    for box, class_id, score in zip(boxes, class_ids, scores):
        print(f"Box: {box}, Class ID: {class_id}, Score: {score}")

        # Extract coordinates
        x1, y1, x2, y2 = box
        color = get_class_color(class_id)
        label = f"{labels[class_id]}: {score:.2f}"
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, font_size, 1)
        cv.rectangle(img, (x1, y1 - int(1.3 * text_height)), (x1 + text_width, y1), color, -1)
        cv.putText(img, label, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1, cv.LINE_AA)

def process_image(img, font_size):
    if model is None:
        return "Model not loaded!", img

    # Convert PIL Image to OpenCV format
    original_img = np.array(img)
    img = cv.cvtColor(original_img, cv.COLOR_RGB2BGR)

    # Store original dimensions
    original_height, original_width = img.shape[:2]

    # Resize and preprocess the image for model input
    img_resized = cv.resize(img, (input_width, input_height))
    scaling_factor_width = original_width / input_width
    scaling_factor_height = original_height / input_height
    im = img_resized.transpose((2, 0, 1))
    im = np.expand_dims(im, 0)
    im = im.astype(np.float32)
    im /= 255.0

    # Run inference
    outname = [i.name for i in model.get_outputs()]
    print(outname)
    num_dets, det_boxes, det_scores, det_classes = model.run(outname, {'images': im})

    # Convert outputs to flat lists and filter
    det_boxes = det_boxes[0]
    det_scores = det_scores[0]
    det_classes = det_classes[0]

    valid_indices = [i for i, cls_id in enumerate(det_classes) if cls_id != -1]
    det_boxes = [det_boxes[i] for i in valid_indices]
    det_classes = [det_classes[i] for i in valid_indices]
    det_scores = [det_scores[i] for i in valid_indices]

    # Scale bounding boxes back to original image dimensions
    det_boxes_scaled = [
        [int(coord * scaling_factor_width if i % 2 == 0 else coord * scaling_factor_height) 
         for i, coord in enumerate(box)] for box in det_boxes]

    # Draw bounding boxes and labels on the original image
    draw_boxes(original_img, det_boxes_scaled, det_classes, det_scores, font_size)

    return original_img

def load_model(model_path, width, height, provider):
    global model, input_width, input_height
    input_width = int(width)
    input_height = int(height)
    providers = {
        'TensorRT': 'TensorrtExecutionProvider',
        'CUDA': 'CUDAExecutionProvider',
        'CPU': 'CPUExecutionProvider'
    }
    print(f"Selected provider: {provider}")
    if provider in providers:
        chosen_provider = providers[provider]
        if chosen_provider in ort.get_available_providers():
            model = ort.InferenceSession(model_path, providers=[chosen_provider])
            return f"Model loaded with {provider} at dimensions {width}x{height}"
        else:
            return f"{provider} not available. Model not loaded."
    else:
        return "Invalid provider selected. Model not loaded."

def update_labels(*args):
    global labels
    labels = list(args)
    print(labels)
    #get current directory and join it with the path to the labels file)
    print(os.path.join(os.getcwd(), "data/id_labels.ini"))
    save_labels(os.path.join(os.getcwd(), "data/id_labels.ini"))
    return "Labels updated successfully!"

def save_labels(path):
    config = configparser.ConfigParser()
    config['Labels'] = {f'Class{i}': label for i, label in enumerate(labels)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as configfile:
        config.write(configfile)

def load_labels(path):
    config = configparser.ConfigParser()
    config.read(path)
    global labels
    if 'Labels' in config:
        labels = [config['Labels'].get(f'Class{i}', '') for i in range(len(config['Labels']))]
    else:
        labels = []
    return labels