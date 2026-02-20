# Project Saiyan AR  
### Real-Time Gesture-Controlled Face Transformation & AR Effects Engine

A high-performance, modular Augmented Reality (AR) engine built in Python that performs real-time gesture recognition, landmark-based face transformation, and cinematic energy effects using computer vision.

---

## 🚀 Overview

Project Saiyan AR is a real-time computer vision system that:

- Detects facial landmarks (468-point mesh)
- Detects and tracks hand gestures
- Performs geometric face warping
- Applies seamless blending for realistic transformation
- Triggers cinematic energy effects based on gestures
- Runs at real-time frame rates (30+ FPS)

The system is designed with modular architecture and professional-level visual quality.

---

## ✨ Key Features

- 🔍 Real-time Face Tracking (468 landmarks)
- 🖐 Real-time Hand Tracking (21 landmarks per hand)
- 🎭 Landmark-Based Face Transformation Engine
- 🎬 Smooth Animated Transformation Transitions
- 💥 Gesture-Triggered Energy Effects (Kamehameha-style)
- 🧠 State Machine-Based Gesture Recognition
- ⚡ Real-time Performance Optimized (30–45 FPS)
- 🧩 Modular and Extensible Architecture

---

## 🏗 System Architecture

```
Camera Input
↓
Frame Processor
↓
Face Tracker ─── Hand Tracker
↓
Gesture Engine
↓
Face Swap Engine ─── Effects Engine
↓
Renderer
```


---

## 🛠 Technology Stack

- Python 3.10+
- OpenCV
- MediaPipe
- NumPy
- (Optional) PyTorch for AI extensions

---

## 📂 Project Structure

```
project-saiyan-ar/
│
├── main.py
├── camera.py
├── face_tracker.py
├── hand_tracker.py
├── gesture_engine.py
├── face_swap_engine.py
├── effects_engine.py
├── utils.py
├── assets/
│ ├── reference_face.png
│ ├── energy_effects/
│
└── requirements.txt
```


---

## 🧠 How It Works

### 1️⃣ Face Tracking
Uses MediaPipe Face Mesh to extract 468 facial landmarks for precise geometry mapping.

### 2️⃣ Hand Tracking
Detects 21 landmarks per hand to analyze gesture positions and velocity.

### 3️⃣ Gesture Recognition
A frame-based state machine detects:

- Face swipe gesture → triggers transformation
- Dual-hand energy pose → triggers energy effect

### 4️⃣ Face Transformation Engine
- Delaunay triangulation
- Affine transformation per triangle
- Seamless blending (Poisson blending)
- Lighting and color correction

### 5️⃣ Effects Engine
- Particle-based energy ball
- Additive blending glow
- Motion blur simulation
- Animated beam rendering

---

## 🖥 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Mohid-Abbas/project-saiyan-ar.git
cd project-saiyan-ar
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Running the Project

```bash
python main.py
```

Press q to exit.

### 🎯 Functional Requirements

    - Real-time face detection (<100ms latency)
    - Real-time hand detection
    - Gesture-triggered transformation
    - Seamless face blending
    - ≥ 30 FPS performance

### 📊 Performance Targets

    - Resolution: 720p minimum
    - Frame Rate: 30–45 FPS
    - Gesture Detection Accuracy: > 90%
    - Stable landmark smoothing

### ⚠️ Legal Notice

This project is inspired by anime-style transformations.
For public deployment or distribution, use original or royalty-free assets.
Do not use copyrighted characters or artwork without permission.

### 🔮 Future Improvements

    - Multiple transformation modes
    - Voice-trigger activation
    - Real-time diffusion style transfer
    - GPU acceleration (CUDA)
    - Web deployment (WebRTC)
    - Mobile port
    - Unity integration

### 📌 Project Goals

This project demonstrates:

    - Computer Vision
    - Real-Time Systems
    - Geometric Image Processing
    - Human-Computer Interaction
    - AR Rendering Techniques
    - Gesture Recognition Architecture

### 👨‍💻 Author

Muhammad Mohid Abbas
Computer Vision & AI Enthusiast

⭐ If You Like This Project

Give it a star ⭐ and feel free to contribute!
