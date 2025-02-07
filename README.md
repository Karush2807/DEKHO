# **🔍 Criminal Tracking & Geolocation Using CCTV 🎥🚨**  

## **🚀 Overview**  
This project is an **AI-powered real-time tracking system** that detects and tracks individuals across multiple CCTV cameras, **mapping their real-world locations** using **homography-based geolocation**.  

✅ **Law enforcement** can use it to **track criminals**  
✅ **Urban authorities** can analyze **pedestrian movement**  
✅ **Industries** can **monitor worker safety compliance**  

### **Key Features:**  
- 🏃‍♂️ **Real-time person detection & tracking** (YOLOv8 + DeepSORT)  
- 🌍 **Geolocation mapping** from CCTV pixels → real-world coordinates  
- 🚨 **Automated alert system** for restricted areas & unauthorized access  
- 📡 **Multi-camera support** with homography-based tracking  
- 📊 **Live dashboard for visualization** (Streamlit + Leaflet.js)  

---

## **🛠️ Tech Stack**  
🔹 **Deep Learning:** YOLOv8 (Object Detection) + DeepSORT (Tracking)  
🔹 **Computer Vision:** OpenCV, Homography Mapping  
🔹 **Backend:** FastAPI (for API-based tracking system)  
🔹 **Frontend:** Streamlit + Leaflet.js (for live tracking visualization)  
🔹 **Database:** Firebase / PostgreSQL (for storing movement logs)  

---
## **🗂️ Folder Structure**  
```
📦 Criminal-Tracking-CCTV
│-- 📂 backend/               # Backend services
│   │-- 📂 api/               # API endpoints
│   │   │-- auth.py           # Authentication module
│   │   │-- main.py           # Main API entry point
│   │   │-- tracking.py       # Tracking API
│   │-- 📂 database/          # Database management
│   │   │-- db.py             # Database connection
│   │   │-- models.py         # Data models
│   │-- 📂 models/            # AI/ML models
│   │   │-- 📂 deepsort/       # DeepSORT tracking model
│   │   │-- 📂 reid/          # Re-identification model
│   │   │-- 📂 yolov5/        # YOLOv5 detection model
│   │-- 📂 utils/             # Utility functions
│   │   │-- alert_system.py   # Alert & notification system
│   │   │-- camera_system.py  # CCTV/Webcam feed management
│   │   │-- detect_tracking.py# YOLOv8 + DeepSORT for tracking
│   │   │-- homography.py     # Homography mapping for geolocation
│-- 📂 frontend/              # Frontend visualization (Streamlit)
│-- .env                     # Environment variables
│-- .gitignore               # Git ignore file
│-- config.py                # Configuration settings
│-- docker-compose.yml       # Docker setup
│-- README.md                # Project Documentation
│-- requirements.txt         # Dependencies
```

---

## **🚀 How It Works**  

### **📌 Step 1: Capture CCTV Footage 🎥**  
📍 Uses **IP Cameras / Webcams / RTSP Streams** as input.  
```bash
python camera_system.py
```

### **📌 Step 2: Detect & Track Individuals 🏃‍♂️**  
📍 Runs **YOLOv8** for person detection & **DeepSORT** for tracking.  
```bash
python detect_tracking.py
```

### **📌 Step 3: Convert to Real-World Coordinates 🌍**  
📍 Maps CCTV pixels to **geolocation** using **homography transformation**.  
```bash
python homography.py
```

### **📌 Step 4: Trigger Alerts 🚨**  
📍 Detects unauthorized individuals & sends alerts.  
```bash
python alert_system.py
```

### **📌 Step 5: Live Tracking Dashboard 📊**  
📍 Visualizes movements on an **interactive map**.  
```bash
streamlit run dashboard.py
```

---

## **⚡ Live Demo Setup**  
🚀 **Don’t have CCTV? No problem!**  
**Use a webcam as an alternative:**  
```bash
python main.py --camera 0
```
📍 **To simulate a real CCTV environment, use:**  
```bash
python main.py --camera "rtsp://your-cctv-url"
```

---

## **🔗 Future Improvements**  
✅ **Multi-camera synchronization** across different locations  
✅ **Advanced re-identification models** for better tracking across angles  
✅ **Integration with law enforcement databases**  

---

## **👨‍💻 Team & Contributors**  
💡 **Project Lead:** *Your Name*  
🤖 **AI/ML Engineer:** *Your Name*  
🌍 **Computer Vision Expert:** *Your Name*  

---

## **📜 License**  
This project is **open-source** under the **MIT License**.  

---

## **⭐ Hackathon Checklist**  
✅ **Clear problem statement**  
✅ **Live demo-ready prototype**  
✅ **Optimized real-time performance**  
✅ **Well-documented code & workflow**  
