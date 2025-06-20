from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import base64
from PIL import Image
import io

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = "runs/detect_train/weights/best.pt"  
model = None

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            print(f"[INFO] Loaded trained model from {MODEL_PATH}")
        else:
            # Fallback to base model for testing
            model = YOLO("yolov8n.pt")
            print("[WARNING] Using base YOLOv8n model. Train your model first!")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_class_descriptions():
    descriptions = {
        'Anti Lock Braking System': 'ABS Light - Anti-lock Braking System issue. Brakes still work but ABS may not.',
        'Braking System Issue': 'Brake System Warning - Brake system issue detected. Check immediately!',
        'Charging System Issue': 'Battery Warning - Charging system problem. Battery may be failing.',
        'Check Engine': 'Check Engine Light - Your engine needs attention. Check with a mechanic soon.',
        'Electronic Stability Problem -ESP-': 'Electronic Stability Control issue. Your car may lose grip or traction control.',
        'Engine Overheating Warning Light': 'Engine Temperature Warning - Engine is overheating. Stop and let cool down.',
        'Low Engine Oil Warning Light': 'Oil Pressure Warning - Low oil pressure detected. Stop driving immediately!',
        'Low Tire Pressure Warning Light': 'Low Tire Pressure - Inflate tires to recommended pressure.',
        'Master warning light': 'Master Warning Light - A general warning has been triggered. Check the vehicle’s system.',
        'SRS-Airbag': 'Airbag Warning - Airbag system malfunction. Safety feature may not work.'
    }
    return descriptions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Run prediction
        results = model(filepath)
        
        # Process results
        predictions = []
        class_descriptions = get_class_descriptions()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class name and confidence
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Get description
                    description = class_descriptions.get(class_name, f"Dashboard warning: {class_name}")
                    
                    predictions.append({
                        'class': class_name,
                        'confidence': round(confidence * 100, 1),
                        'description': description,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
        
        # Convert image to base64 for display
        with open(filepath, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'predictions': predictions,
            'image': img_base64,
            'total_detections': len(predictions)
        })
        
    except Exception as e:
        # Clean up file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("[INFO] Starting Car Dashboard Icon Detector Web App...")
    
    # Load the model
    if not load_model():
        print("[ERROR] Could not load model. Please train your model first!")
        exit(1)
    
    print("[INFO] Model loaded successfully!")
    print("[INFO] Starting Flask server...")
    print("[INFO] Access the app at: http://localhost:5050")
    
    app.run(debug=True, host='0.0.0.0', port=5050)