# CompV - Professional Object Detection System

Clean, organized project structure for real-time object detection with web interface.

## 📁 Project Structure

```
CompV/
├── 📂 modules/              # Core Python modules
│   ├── __init__.py
│   ├── statistics.py        # Detection statistics & analytics
│   ├── tracker.py           # Centroid-based object tracking
│   ├── zone_detector.py     # Zone/ROI detection
│   └── alert_manager.py     # Email & webhook alerts
│
├── 📂 models/               # YOLO model files
│   ├── yolov3.weights       # YOLOv3 weights (237 MB)
│   ├── yolov3.cfg           # YOLOv3 config
│   ├── yolov8n.pt           # YOLOv8 nano model (6 MB)
│   └── coco.names           # Class names (80 classes)
│
├── 📂 templates/            # Web UI templates
│   └── index.html           # Dashboard interface
│
├── 📂 static/               # Web assets
│   ├── css/
│   │   └── style.css        # Dashboard styling
│   └── js/
│       └── dashboard.js     # Client-side logic
│
├── 📂 docs/                 # Documentation
│   └── README.md            # Full documentation
│
├── 📂 outputs/              # Generated outputs
│   ├── *.mp4                # Processed videos
│   └── *.csv                # Statistics exports
│
├── 🐍 web_app.py            # Flask web application
├── 🐍 object_detection_yolov8.py  # YOLOv8 CLI
├── 🐍 object_detection.py   # YOLOv3 CLI (legacy)
├── ⚙️ config.json           # System configuration
├── 📋 requirements.txt      # Python dependencies
├── 🚫 .gitignore            # Git ignore rules
└── 📖 PROJECT_STRUCTURE.md  # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Web Dashboard
```bash
python web_app.py
# Open browser: http://localhost:5000
```

### 3. Or Use CLI
```bash
# Process video
python object_detection_yolov8.py --input video.mp4

# Use camera
python object_detection_yolov8.py --camera 0 --tracking --stats-output stats.csv
```

## 📦 What's in Each Folder

### `modules/` - Core Functionality
- **statistics.py**: Track detection counts, confidence scores, export to CSV/JSON
- **tracker.py**: Assign persistent IDs to detected objects
- **zone_detector.py**: Define detection zones, count objects in areas
- **alert_manager.py**: Send alerts via email or webhook

### `models/` - YOLO Models
- **yolov3.weights**: Original YOLO model (slower, legacy)
- **yolov8n.pt**: Modern YOLO nano (faster, recommended)
- **coco.names**: 80 object classes (person, car, dog, etc.)

### `templates/` & `static/` - Web Interface
- Responsive Bootstrap 5 dashboard
- Real-time video streaming
- Live statistics updates
- Configuration controls

### `outputs/` - Results
- Processed videos with detections
- Statistics CSV files
- Auto-generated outputs

## 🎯 Features

- ✅ Real-time object detection (YOLOv8)
- ✅ Web dashboard with live streaming
- ✅ Object tracking with persistent IDs
- ✅ Zone-based detection (ROI)
- ✅ Email & webhook alerts
- ✅ Statistics export (CSV/JSON)
- ✅ GPU acceleration ready
- ✅ Video file & camera support

## ⚙️ Configuration

Edit `config.json` to customize:
- Model settings (confidence, IOU)
- Video source (camera/file)
- Features (tracking, zones, alerts)
- Web server (host, port)
- Alert settings (email, webhook)

## 📊 Performance

| Mode | FPS | Use Case |
|------|-----|----------|
| CPU | 6-8 | Testing, demo |
| GPU (CUDA) | 80-120 | Production |

## 🔧 Maintenance

All temporary files and outputs go to `outputs/` folder.
Model files stay in `models/` folder.
Clean structure, easy to navigate!

---

**Built with:** Python, Flask, OpenCV, YOLOv8, Bootstrap
