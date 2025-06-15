from ultralytics import YOLO

# Path to the dataset config file
DATA_YAML = "data/data.yaml"

# We can use any of these base models: yolov8n.pt, yolov8s.pt, yolov8m.pt
BASE_MODEL = "yolov8n.pt"

# Number of training epochs
EPOCHS = 50

# Image size
IMG_SIZE = 640

def train():
    print("[INFO] Initializing YOLO model...")
    # Initialization of YOLO model with pre-trained weights 

    model = YOLO(BASE_MODEL)

    print(f"[INFO] Training model on: {DATA_YAML}")
    # Starts training with parameters from data.yaml
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project="runs",
        name="detect_train",
        exist_ok=True  # allows retraining without error
    )

    print("[INFO] Training complete. Best model saved to:")
    print("→ runs/detect_train/weights/best.pt")

if __name__ == "__main__":
    train()
