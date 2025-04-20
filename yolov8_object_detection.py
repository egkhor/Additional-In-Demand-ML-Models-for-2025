from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # YOLOv8 nano model (lightweight)

# Load an example image (replace with your image)
image = cv2.imread("example_image.jpg")

# Perform object detection
results = model(image)

# Draw bounding boxes and labels on the image
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
    classes = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
    names = result.names  # Class names

    for box, conf, cls in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = box
        label = f"{names[cls]} {conf:.2f}"
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the result
cv2.imwrite("yolov8_detection_result.jpg", image)
print("Detection result saved as yolov8_detection_result.jpg")

# Save the model (YOLOv8 saves weights automatically during training, here we just export)
model.export(format="onnx")
print("Model exported as yolov8n.onnx")
