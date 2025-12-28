# CompV Project Structure

```
CompV/
├── modules/                      # Python modules
│   ├── __init__.py
│   ├── statistics.py            # Detection statistics
│   ├── tracker.py               # Object tracking
│   ├── zone_detector.py         # Zone/ROI detection
│   └── alert_manager.py         # Alert system
│
├── templates/                    # HTML templates
│   └── index.html               # Web dashboard
│
├── static/                       # Web assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── dashboard.js
│
├── docs/                         # Documentation
│   └── README.md
│
├── outputs/                      # Output files
│   ├── *.mp4                    # Video outputs
│   └── *.csv                    # Statistics exports
│
├── web_app.py                    # Web application (Flask)
├── object_detection.py           # YOLOv3 CLI (legacy)
├── object_detection_yolov8.py    # YOLOv8 CLI
├── config.json                   # Configuration
├── requirements.txt              # Dependencies
├── .gitignore                    # Git ignore rules
│
└── Model files (download separately):
    ├── yolov3.weights
    ├── yolov3.cfg
    ├── yolov8n.pt
    └── coco.names
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run web dashboard
python web_app.py

# Or run CLI
python object_detection_yolov8.py --input video.mp4
```
