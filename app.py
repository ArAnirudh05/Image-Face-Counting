import streamlit as st

# Set page config
st.set_page_config(page_title="Object Detection App", layout="wide")

import cv2
import numpy as np
from PIL import Image
import io

# Load YOLO model and COCO class names
@st.cache_resource
def load_model():
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    layer_names = net.getUnconnectedOutLayersNames()
    return net, layer_names

net, layer_names = load_model()

# COCO dataset class names
COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
               'hair drier', 'toothbrush']

def detect_objects(image, conf_thresh=0.5, nms_thresh=0.4):
    H, W = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (608,608)), 1/255.0, (608,608), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    boxes, confidences, class_ids = [], [], []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            confidence = max(scores)
            if confidence > conf_thresh:
                class_id = np.argmax(scores)
                cx, cy, w, h = detection[0]*W, detection[1]*H, detection[2]*W, detection[3]*H
                x = int(cx - w/2)
                y = int(cy - h/2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

    final_boxes, final_classes = [], []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_classes.append(class_ids[i])

    return final_boxes, final_classes

# Add a title
st.title("Object Detection Application")

# Add file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Process the image
    boxes, classes = detect_objects(img)
    
    # Draw bounding boxes
    for (x, y, w, h), class_id in zip(boxes, classes):
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        
        # Add label
        class_name = COCO_CLASSES[class_id]
        label = f"{class_name}"
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Count objects by class
    class_counts = {}
    for class_id in classes:
        class_name = COCO_CLASSES[class_id]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    with col2:
        st.subheader("Detected Objects")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Display counts
    st.subheader("Detection Results")
    col3, col4 = st.columns(2)
    
    with col3:
        st.metric("Total Objects Detected", len(boxes))
    
    with col4:
        st.write("Objects Detected:")
        for class_name, count in class_counts.items():
            st.write(f"- {class_name}: {count}")