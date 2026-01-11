from ultralytics import YOLO
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

print("🚀 Starting YOLOv8 training...")

# Load pretrained model
model = YOLO('yolov8n-seg.pt')

# Train model
results = model.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='franka_grasp_v1',
    project='../runs/detect'
)

print("✅ Training completed! Weights saved at runs/detect/franka_grasp_v1/weights/best.pt")
