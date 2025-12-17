from ultralytics import YOLO
import os

# Garante que estamos no diretório certo
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

print("🚀 Iniciando Treinamento YOLOv8...")

# Carrega modelo pré-treinado
model = YOLO('yolov8n.pt') 

# Treina
# Certifique-se que no seu data.yaml o path está correto ou absoluto!
results = model.train(
    data='data.yaml',
    epochs=50,        
    imgsz=640,
    batch=16,
    name='franka_grasp_v1',
    project='../runs/detect' # Salva na pasta runs fora de training
)

print("✅ Treino concluído! Pesos salvos em runs/detect/franka_grasp_v1/weights/best.pt")