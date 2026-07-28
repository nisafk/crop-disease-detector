# 🌿 CropAI — Crop Disease Detector

An AI-powered web application that detects crop diseases from leaf photos using Computer Vision and Deep Learning.

## 🎯 What it Does
Upload a photo of a diseased leaf → AI identifies the disease → Get treatment advice instantly.

**Tech Stack:** PyTorch (ResNet50) + FastAPI + React  
**Dataset:** PlantVillage (54,306 images, 38 classes)  
**Accuracy:** ~94% on validation set

## 🗂️ Project Structure
```
crop-disease-detector/
├── model/
│   ├── train.py            # Train the CNN model (run on Google Colab)
│   ├── disease_info.json   # Disease → treatment database
│   └── best_model.pth      # Trained model weights (after training)
├── backend/
│   ├── main.py             # FastAPI server (/predict endpoint)
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   ├── App.css         # Styles
│   │   └── index.css       # Global styles
│   └── .env                # API URL config
└── README.md
```

## 🚀 How to Run Locally

### Step 1: Train the Model (Google Colab)
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Upload `model/train.py`
4. Download PlantVillage dataset from Kaggle
5. Run training (~1 hour)
6. Download `best_model.pth` and `class_names.json` → put in `model/` folder

### Step 2: Start Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Visit http://localhost:8000/docs to see the API documentation

### Step 3: Start Frontend
```bash
cd frontend
npm install
npm start
```
Visit http://localhost:3000

## 🌐 Deploy to Production

### Backend → Render.com
1. Push code to GitHub
2. render.com → New Web Service → Connect repo
3. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variable if needed

### Frontend → Vercel
1. vercel.com → Import GitHub repo
2. Set environment variable: `REACT_APP_API_URL=https://your-backend.onrender.com`
3. Deploy!

## 📊 Model Details
- **Architecture:** ResNet50 (Transfer Learning)
- **Pretrained on:** ImageNet (14M images)
- **Fine-tuned on:** PlantVillage dataset
- **Input size:** 224×224 RGB images
- **Output:** 38 disease class probabilities

## 🌾 Supported Crops & Diseases
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato — covering 38 disease classes.

## 👨‍💻 Built By
[Nisha fatima] — 2nd Year AIML Student  
[Nisha fatima kabuli] | [nisafk]

## 📄 License
MIT License
