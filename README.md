🦴 OsteoScan — AI-Powered Bone Cancer Detection System
<p align="center"> <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/> <img src="https://img.shields.io/badge/Flask-Web%20Framework-black?style=for-the-badge&logo=flask"/> <img src="https://img.shields.io/badge/YOLOv8-Deep%20Learning-red?style=for-the-badge"/> <img src="https://img.shields.io/badge/AI-Healthcare-green?style=for-the-badge"/> <img src="https://img.shields.io/badge/Open%20Source-Project-orange?style=for-the-badge"/> </p> <h3 align="center"> 🧠 Intelligent Bone Cancer Detection using Deep Learning & Medical Imaging </h3> <p align="center"> Built with <b>YOLOv8</b>, <b>Flask</b>, and <b>AI-powered Medical Imaging</b> to assist in bone cancer prediction from X-ray scans. </p>
📌 Overview

OsteoScan is an AI-powered healthcare application designed to analyze bone X-ray images and classify them into:

✅ Cancer Detected
✅ Normal Bone Condition

The system leverages a custom-trained YOLOv8 Deep Learning Model integrated with a Flask-based web application for real-time prediction and visualization.

This project demonstrates the practical application of Artificial Intelligence in healthcare diagnostics, enabling intelligent medical image analysis with confidence scoring and user-friendly interaction.

✨ Key Features
🔬 AI-Based Medical Imaging
Detects bone cancer patterns from X-ray scans
Uses trained YOLOv8 model (best.pt)
Provides prediction confidence percentage
Fast and lightweight inference pipeline
🌐 Full Stack Web Application
Flask backend server
Interactive frontend UI
Real-time image upload & prediction
📷 Smart Image Processing

Supported image formats:

JPG
JPEG
PNG
BMP
WEBP
📊 Prediction Analytics

The application displays:

Prediction Label
Confidence Score
Class Probabilities
Uploaded Image Preview
⚡ Optimized Workflow
Automatic upload handling
Runtime upload directory creation
Lightweight REST API pipeline
🛡️ Health Monitoring
/health endpoint checks server & model status
🏗️ System Architecture
bone_cancer_detection/
│
├── app.py
├── users.json
├── server.log
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html
│
├── static/
│   └── uploads/
│
├── weights/
│   └── best.pt
│
└── scratch/
🔄 Application Workflow
🧠 AI Model Workflow
⚙️ Technology Stack
Technology	Purpose
Python	Core Programming
Flask	Backend Framework
YOLOv8	Deep Learning Model
HTML/CSS/JavaScript	Frontend UI
Pillow	Image Processing
Ultralytics	YOLOv8 Implementation
🚀 Installation Guide
1️⃣ Clone Repository
git clone https://github.com/Prachipandey28/bone_cancer_detection.git

cd bone_cancer_detection
2️⃣ Install Dependencies
pip install flask ultralytics pillow

OR

pip install -r requirements.txt
3️⃣ Add Model Weights

Place your trained model inside:

weights/best.pt
4️⃣ Run Application
python app.py

Server will start at:

http://localhost:5000
🌐 REST API Endpoints
Method	Endpoint	Description
GET	/	Load Frontend UI
POST	/predict	Upload Image & Get Prediction
GET	/health	Check Model Status
📤 Prediction API Example
Request
POST /predict
Content-Type: multipart/form-data
Form Data
file=<image_file>
Response
{
  "status": "success",
  "prediction": "cancer",
  "confidence": 94.37,
  "class_probabilities": {
    "cancer": 0.9437,
    "normal": 0.0563
  }
}
📸 Prediction Flow Diagram
📊 Supported File Formats
Format	Supported
JPG	✅
JPEG	✅
PNG	✅
BMP	✅
WEBP	✅
🔒 Disclaimer

⚠️ This project is intended for educational and research purposes only.

It should NOT be used as a substitute for:

Professional medical diagnosis
Clinical decision-making
Healthcare consultation
🚀 Future Enhancements
📈 Improve model accuracy with larger datasets
🩺 Multi-class tumor detection
☁️ Cloud deployment
👤 Authentication system
📊 Prediction history dashboard
🧠 Explainable AI visualization
📱 Better mobile responsiveness
👩‍💻 Developer
👤 Prachi Pandey

🎓 B.Tech — Artificial Intelligence & Data Science
💡 Passionate about:

Artificial Intelligence
Healthcare Technology
Deep Learning
Medical Imaging
🙏 Special Thanks

Special thanks to my friend Mohit Sharma for helping in the model training process and supporting the development of this project.

🔗 GitHub:
Mohit Sharma GitHub

🤝 Contributing

Contributions, feature suggestions, and improvements are welcome.

Steps to Contribute
1. Fork the repository
2. Create a new branch
3. Make changes
4. Commit your updates
5. Push the branch
6. Open a Pull Request
⭐ Support This Project

If you found this project useful:

⭐ Star the repository
🍴 Fork the project
📢 Share with others
📜 License

This project is open-source and available under the MIT License.

💙 Connect With Me
🔗 GitHub Repository

OsteoScan Repository

<p align="center"> <b>🦴 AI for Healthcare • Deep Learning • Medical Imaging • Innovation 🚀</b> </p>

Based on your uploaded project content.
