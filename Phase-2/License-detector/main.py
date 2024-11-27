import numpy as np
import cv2
from skimage import io
import plotly.express as px
import pytesseract as pt
import os

# Set the Tesseract executable path (adjust if running locally)
pt.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Homebrew installation path on macOS

# Define input width and height for YOLO
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# Function to get detections from YOLO model
def get_detections(img, net):
    image = img.copy()
    row, col, d = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    
    return input_image, detections

# Function for non-maximum suppression
def non_maximum_supression(input_image, detections):
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]  # Confidence of detection
        if confidence > 0.4:
            class_score = row[5]  # Class probability score
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)
    
    return boxes_np, confidences_np, index

# Function to draw detections and extract text
def drawings(image, boxes_np, confidences_np, index):
    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)
        license_text = extract_text(image, boxes_np[ind])

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 0, 255), -1)
        cv2.rectangle(image, (x, y + h), (x + w, y + h + 25), (0, 0, 0), -1)

        cv2.putText(image, conf_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(image, license_text, (x, y + h + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    return image

# Function to get YOLO predictions
def yolo_predictions(img, net):
    input_image, detections = get_detections(img, net)
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    result_img = drawings(img, boxes_np, confidences_np, index)
    return result_img

# Function to extract text using Tesseract OCR
def extract_text(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    
    if 0 in roi.shape:
        return 'no number'
    else:
        text = pt.image_to_string(roi)
        text = text.strip()
        
        return text

# Load YOLO model
net = cv2.dnn.readNetFromONNX('./yolov5/runs/train/Model/weights/best.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Function to process an image
def process_image(image_path):
    img = io.imread(image_path)
    results = yolo_predictions(img, net)
    fig = px.imshow(results)
    fig.update_layout(width=700, height=400, margin=dict(l=10, r=10, b=10, t=10))
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.show()

# Function to process a video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Unable to read video')
            break
        
        results = yolo_predictions(frame, net)
        cv2.namedWindow('YOLO', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('YOLO', results)
        
        if cv2.waitKey(30) == 27:  # Escape key to stop
            break

    cv2.destroyAllWindows()
    cap.release()

# Function to process live camera feed
def process_live_camera():
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Unable to access the camera')
            break
        
        results = yolo_predictions(frame, net)
        cv2.namedWindow('YOLO Live', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('YOLO Live', results)
        
        if cv2.waitKey(30) == 27:  # Escape key to stop
            break

    cv2.destroyAllWindows()
    cap.release()

# Example usage
if __name__ == "__main__":
    # Choose the mode: 'image', 'video', or 'live'
    mode = 'live'  # Change to 'image' or 'video' as needed

    if mode == 'image':
        image_path = 'TEST.jpeg'
        if os.path.exists(image_path):
            process_image(image_path)
        else:
            print(f"Image not found at {image_path}")

    elif mode == 'video':
        video_path = '../input/number-plate-detection/TEST/TEST.mp4'
        if os.path.exists(video_path):
            process_video(video_path)
        else:
            print(f"Video not found at {video_path}")
    
    elif mode == 'live':
        process_live_camera()

    else:
        print("Invalid mode selected. Choose 'image', 'video', or 'live'.")
