🦴 OsteoScan — AI-Powered Bone Cancer Detection System
<p align="center"> <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" /> <img src="https://img.shields.io/badge/Flask-Web%20Framework-black?style=for-the-badge&logo=flask" /> <img src="https://img.shields.io/badge/YOLOv8-Deep%20Learning-red?style=for-the-badge" /> <img src="https://img.shields.io/badge/AI-Medical%20Imaging-green?style=for-the-badge" /> </p> <p align="center"> <b>Deep Learning-Based Medical Imaging Platform for Bone Cancer Detection using YOLOv8 and Flask</b> </p> <p align="center"> <i>Built for AI Healthcare Research, Medical Imaging Analysis, and Intelligent Diagnostic Assistance</i> </p>
📌 Overview

OsteoScan is an AI-powered web application designed to analyze bone X-ray images and predict whether the scan indicates:

✅ Cancer Detected
✅ Normal Bone Condition

The project combines:

🧠 YOLOv8 Deep Learning Model
🌐 Flask Backend
🎨 Interactive Frontend UI
📷 Real-Time Image Upload & Prediction
📊 Confidence Score Visualization

This project was developed to explore the practical application of Artificial Intelligence in healthcare diagnostics using deep learning techniques.

✨ Key Features
🔬 AI-Powered Medical Imaging
Detects bone cancer patterns from X-ray scans
Uses trained YOLOv8 model weights (best.pt)
Generates prediction with confidence percentage
🌐 Full Stack Architecture
Flask backend server
Responsive frontend interface
Real-time image upload and prediction system
📷 Smart Image Processing

Supported formats:

JPG
JPEG
PNG
BMP
WEBP
📊 Prediction Output

The application provides:

Prediction label
Confidence score
Class probabilities
Image preview support
⚡ Optimized Prediction Pipeline
Automatic upload handling
Runtime upload folder creation
Lightweight API workflow
🛡️ Health Monitoring Endpoint
/health endpoint checks model loading status
🏗️ System Architecture
bone_cancer_detection/
│
├── app.py                 # Flask backend server
├── users.json             # User data
├── server.log             # Runtime logs
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html         # Frontend UI
│
├── static/
│   └── uploads/           # Uploaded images
│
├── weights/
│   └── best.pt            # YOLOv8 trained model
│
└── scratch/
⚙️ Technology Stack
Technology	Purpose
Python	Core programming language
Flask	Backend framework
YOLOv8	Deep learning model
HTML/CSS/JavaScript	Frontend UI
Pillow	Image handling
Ultralytics	YOLOv8 implementation
🚀 Installation & Deployment
1️⃣ Clone the Repository
git clone https://github.com/Prachipandey28/bone_cancer_detection.git

cd bone_cancer_detection
2️⃣ Install Dependencies
pip install flask ultralytics pillow

Or using requirements file:

pip install -r requirements.txt
3️⃣ Add Model Weights

Place your trained YOLOv8 model inside:

weights/best.pt
4️⃣ Run the Application
python app.py

Server starts at:

http://localhost:5000
🌐 REST API Endpoints
Method	Endpoint	Description
GET	/	Load frontend UI
POST	/predict	Upload image & get prediction
GET	/health	Check model status
📤 Prediction API Example
Request
POST /predict
Content-Type: multipart/form-data
Form Data
file=<image_file>
Response Example
{
  "status": "success",
  "prediction": "cancer",
  "confidence": 94.37,
  "class_probabilities": {
    "cancer": 0.9437,
    "normal": 0.0563
  }
}
📸 Application Workflow
Upload X-ray Image
        ↓
Flask Backend Receives Image
        ↓
YOLOv8 Model Processes Scan
        ↓
Prediction Generated
        ↓
Result Returned
        ↓
Displayed on Frontend
🧠 Deep Learning Model

The application uses a custom-trained YOLOv8 classification model for detecting cancer patterns in bone X-ray images.

Model Capabilities
Binary classification:
Cancer
Normal
Fast inference speed
Medical image analysis using deep learning
📊 Supported File Formats
Format	Supported
JPG	✅
JPEG	✅
PNG	✅
BMP	✅
WEBP	✅
🔒 Disclaimer

⚠️ This project is intended for educational and research purposes only.

It should NOT be used as a substitute for professional medical diagnosis or clinical decision-making.

🚀 Future Enhancements
📈 Improve model accuracy with larger datasets
🩺 Multi-class tumor detection
☁️ Cloud deployment
👤 User authentication system
📊 Prediction history dashboard
🧠 Explainable AI visualizations
📱 Better mobile responsiveness
👩‍💻 Developer
👤 Prachi Pandey

B.Tech Artificial Intelligence & Data Science Student
Passionate about AI, Healthcare Technology, and Machine Learning.

🔗 GitHub Repository

bone_cancer_detection Repository

🙏 Acknowledgements

Special thanks to my friend Mohit for helping in model training and supporting this project.

🤝 Contributing

Contributions, feature suggestions, and improvements are welcome.

Steps to contribute:
Fork the repository
Create a new branch
Make changes
Commit your updates
Push the branch
Open a Pull Request
⭐ Project Support

If you found this project useful:

⭐ Star the repository
🍴 Fork the project
📢 Share with others
📜 License

This project is open-source and available under the MIT License.

<p align="center"> <b>🦴 AI for Healthcare • Deep Learning • Medical Imaging • Innovation 🚀</b> </p>
