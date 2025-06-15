# Car Dashboard Warning Light Detector

[](https://www.python.org/downloads/)
[](https://flask.palletsprojects.com/)
[](https://github.com/ultralytics/ultralytics)
[](https://opensource.org/licenses/MIT)

A web-based application that uses a custom-trained **YOLOv8** model to detect and identify car dashboard warning lights from an uploaded image. Get instant explanations for what each light means and what action you should take.



## Features

  - **AI-Powered Detection**: Leverages a YOLOv8 object detection model to accurately identify dashboard icons.
  - **User-Friendly Web Interface**: Clean, modern, and responsive UI built with Flask and modern CSS.
  - **Drag & Drop Upload**: Easily upload images by dragging them onto the page or using a standard file picker.
  - **Instant Analysis**: Get immediate results with bounding boxes drawn on the original image.
  - **Detailed Explanations**: Each detected icon comes with a name, confidence score, and a clear description of its meaning and recommended action.
  - **Easy to Train**: Includes a simple script to train or retrain the model on your own dataset.

## Technologies Used

  - **Backend**:
      - Python 3.9+
      - [Flask](https://flask.palletsprojects.com/): For the web server and API.
      - [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics): For model training and inference.
      - [PyTorch](https://pytorch.org/): As the deep learning framework backbone for YOLO.
      - [OpenCV-Python](https://pypi.org/project/opencv-python/): For image processing.
  - **Frontend**:
      - HTML5
      - CSS3 (with modern features like variables and flexbox)
      - Vanilla JavaScript (for DOM manipulation and API calls)

## Project Structure

```
/car-alert-detector/
│
├── data/                    # Dataset for training and validation
│   ├── train/
│   ├── valid/
│   └── data.yaml            # Dataset configuration file
│
├── runs/                    # Output directory for model training results
│   └── detect_train/
│       └── weights/
│           └── best.pt      # The best trained model weights
│
├── templates/               # Flask HTML templates
│   └── index.html           # Main application page
│
├── uploads/                 # Temporary folder for user uploads
│
├── app.py                   # Main Flask application script
├── car_alert.py             # Script to train the YOLOv8 model
├── requirements.txt         # Python dependencies
└── yolov8n.pt               # Base pre-trained YOLOv8 model
```

## Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

  - Python 3.9 or higher
  - `pip` and `venv`

### 1\. Clone the Repository

```sh
git clone git@github.com:sara-nur/car-alert-detector.git
cd car-alert-detector
```

### 2\. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

  - **Linux/macOS**:

    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

  - **Windows**:

    ```sh
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

### 3\. Install Dependencies

Install all the required Python libraries using the `requirements.txt` file.

```sh
pip install -r requirements.txt
```

## Usage

There are two main steps to using this project: training the model (if needed) and running the web application.

### 1\. Train the Model (Optional)

If you want to train the model on your own dataset or improve the existing one, you can run the training script.

1.  **Prepare your dataset**: Place your annotated images in the `data/train` and `data/valid` folders, following the YOLO format.
2.  **Configure your dataset**: Edit `data/data.yaml` to specify the paths and class names.
3.  **Run the training script**:
    ```sh
    python car_alert.py
    ```
    The best model will be saved as `runs/detect_train/weights/best.pt`. This is the model the Flask app will use automatically.

### 2\. Run the Web Application

To start the Flask server, run the `app.py` script.

```sh
python app.py
```

The application will be available at: **[http://localhost:5050](https://www.google.com/search?q=http://localhost:5050)**

Now you can open your web browser, navigate to the address, and upload an image of a car dashboard to see the detections.

## Dataset

The model is trained on a custom dataset of car dashboard images. The dataset is structured for YOLOv8 and defined in `data/data.yaml`.

  - `data/images/`: Contains the image files (`.jpg`, `.png`).
  - `data/labels/`: Contains the annotation files (`.txt`). Each file corresponds to an image and contains the class and bounding box coordinates for each object in the format: `<class_id> <x_center> <y_center> <width> <height>`.
