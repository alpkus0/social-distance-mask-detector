Social Distance & Mask Detector

Real-time mask detection and social distance monitoring using YOLOv8 and OpenCV.
This project detects whether people wear masks correctly and identifies social distance violations in real-time video streams.

📌 Features

Detects three mask statuses:

✅ with_mask → correctly worn

⚠️ mask_weared_incorrect → incorrectly worn

❌ without_mask → no mask

Highlights each person with a colored bounding box:

🟢 Green → correct mask

🟡 Yellow → incorrect mask

🔴 Red → no mask / social distance violation

Calculates social distance between people and raises warnings if rules are violated

Real-time video output with annotations

Adjustable thresholds for social distancing

📂 Folder Structure

social-distance-mask-detector/
│
├── yolov8_mask_model.pt → Trained custom YOLOv8 model
├── main.py → Real-time detection code
├── convert_xml2yolo.py → Convert XML annotations to YOLO format
├── prepare_data.py → Prepares dataset for training
├── data.yaml → Dataset configuration for YOLOv8
├── requirements.txt → Python dependencies
├── screenshots/ → Example screenshots
├── README.md → This documentation
└── rapor.pdf → Detailed project report (PDF)

⚙️ Installation

Clone this repository
git clone https://github.com/yourusername/social-distance-mask-detector.git
cd social-distance-mask-detector

Install dependencies
pip install -r requirements.txt

Make sure you have a working camera or replace cv2.VideoCapture(0) with a video file path in main.py.

▶️ Usage

Run the main detection script:
python main.py

Press ESC to quit the application.

Bounding boxes and mask status will be displayed on the video feed.

Social distance violations will show warnings in red.

🎯 Training Your Own Model

If you want to retrain the model:

yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=20 imgsz=640 device=0

Parameters:

task=detect → Object detection task

mode=train → Train the model

model=yolov8n.pt → Pretrained YOLOv8 nano model

data=data.yaml → Dataset and class definitions

epochs=20 → Number of training iterations

imgsz=640 → Input image size

device=0 → GPU device

🖼 Screenshots

Place your screenshots in the screenshots/ folder. Example:


📌 Notes

Social distance detection works better with larger datasets; small datasets may cause false alarms.

Model performance may vary depending on lighting, camera angle, and number of people in frame.

📚 References

Dataset used: Kaggle Mask Dataset → https://www.kaggle.com/datasets

YOLOv8 Documentation → https://docs.ultralytics.com/
