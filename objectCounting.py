import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import subprocess
import sys
import time

# Load YOLO model and COCO class names
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
layer_names = net.getUnconnectedOutLayersNames()

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

def detect_objects(image, conf_thresh=0.45, nms_thresh=0.3):
    start_time = time.time()
    
    # Preprocess image
    preprocess_start = time.time()
    H, W = image.shape[:2]
    
    # Enhanced image preprocessing
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (608, 608)), 
                                1/255.0, 
                                (608, 608), 
                                mean=(0.485, 0.456, 0.406),
                                swapRB=True,
                                crop=False)
    preprocess_time = time.time() - preprocess_start

    # Network inference
    inference_start = time.time()
    net.setInput(blob)
    outputs = net.forward(layer_names)
    inference_time = time.time() - inference_start

    # Post-processing
    postprocess_start = time.time()
    boxes, confidences, class_ids = [], [], []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            confidence = max(scores)
            class_id = np.argmax(scores)
            
            # Optimized class-specific confidence thresholds
            if class_id == 0:  # person
                min_conf = 0.25  # Lower threshold specifically for person detection
            elif class_id in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                min_conf = 0.35
            else:
                min_conf = conf_thresh
                
            if confidence > min_conf:
                # Calculate box coordinates
                cx, cy, w, h = detection[0]*W, detection[1]*H, detection[2]*W, detection[3]*H
                x = max(0, int(cx - w/2))
                y = max(0, int(cy - h/2))
                w = min(W - x, int(w))
                h = min(H - y, int(h))
                
                # Filter out invalid boxes
                if w > 0 and h > 0 and w < W and h < H:
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

    final_boxes, final_classes = [], []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_classes.append(class_ids[i])

    # Calculate timing metrics
    postprocess_time = time.time() - postprocess_start
    total_time = time.time() - start_time
    
    # Print performance metrics to terminal
    print("\n=== Performance Metrics ===")
    print(f"Preprocessing time: {preprocess_time:.3f} seconds")
    print(f"Inference time: {inference_time:.3f} seconds")
    print(f"Postprocessing time: {postprocess_time:.3f} seconds")
    print(f"Total processing time: {total_time:.3f} seconds")
    print(f"FPS: {1/total_time:.2f}")
    print("========================\n")

    return final_boxes, final_classes, {
        'preprocess_time': preprocess_time,
        'inference_time': inference_time,
        'postprocess_time': postprocess_time,
        'total_time': total_time,
        'fps': 1/total_time
    }

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection")
        
        # Create GUI components
        self.btn_open = tk.Button(root, text="Open Image", command=self.open_image)
        self.btn_open.pack(pady=10)
        
        # Create frame for counts
        self.counts_frame = tk.Frame(root)
        self.counts_frame.pack(pady=5)
        
        self.total_count_label = tk.Label(self.counts_frame, text="Total Objects Detected: 0")
        self.total_count_label.pack()
        
        self.object_counts_text = tk.Text(self.counts_frame, height=5, width=40)
        self.object_counts_text.pack(pady=5)
        
        self.image_panel = tk.Label(root)
        self.image_panel.pack()

    # Add performance metrics labels
    self.metrics_frame = tk.Frame(root)
    self.metrics_frame.pack(pady=5)
    
    self.fps_label = tk.Label(self.metrics_frame, text="FPS: -")
    self.fps_label.pack()
    
    self.time_label = tk.Label(self.metrics_frame, text="Processing time: -")
    self.time_label.pack()

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        
        img = cv2.imread(file_path)
        if img is not None:
            boxes, classes, metrics = detect_objects(img)
            count = len(boxes)
            
            # Count objects by class
            class_counts = {}
            for class_id in classes:
                class_name = COCO_CLASSES[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Draw bounding boxes with labels
            for (x, y, w, h), class_id in zip(boxes, classes):
                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                
                # Add label
                class_name = COCO_CLASSES[class_id]
                label = f"{class_name}"
                cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Convert to RGB and resize for GUI display
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_display = Image.fromarray(img_display)
            img_display.thumbnail((800, 600))
            
            # Update GUI
            img_tk = ImageTk.PhotoImage(img_display)
            self.image_panel.config(image=img_tk)
            self.image_panel.image = img_tk
            # Update total count
            self.total_count_label.config(text=f"Total Objects Detected: {count}")
            
            # Update object counts text
            self.object_counts_text.delete(1.0, tk.END)
            counts_text = "Objects Detected:\n"
            for class_name, class_count in class_counts.items():
                counts_text += f"{class_name}: {class_count}\n"
            self.object_counts_text.insert(tk.END, counts_text)
            
            # Update performance metrics in GUI
            self.fps_label.config(text=f"FPS: {metrics['fps']:.2f}")
            self.time_label.config(text=f"Processing time: {metrics['total_time']:.3f}s")
        else:
            self.total_count_label.config(text="Failed to load image")
            self.object_counts_text.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()