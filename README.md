# Project Saiyan Augmented Reality (AR)  
### Real-Time Gesture-Controlled Super Saiyan Effects & Cinematic Environment

A high-performance, modular AR engine built in Python that simulates the power of a Super Saiyan. Features real-time gesture recognition, advanced particle physics, and cinematic environmental effects.

---

## 🚀 Overview

Project Saiyan AR is an immersive computer vision system that:

- **Super Saiyan Aura**: Real-time silhouette-hugging electric field (body lightning).
- **Cinematic Kamehameha**: Additive-blended energy sphere with fractal lightning and multi-layer bloom.
- **Continuous Burst**: Gesture-controlled energy blast that fires as long as your palms stay open.
- **Living Environment**: Parallax-scrolled backgrounds with flying rocks and dust debris.
- **Screen Shake (Tremors)**: Realistic camera vibration during energy charge and burst states.

---

## ✨ Key Features

- 🔍 **Real-time Face & Hand Mesh**: 468 face landmarks and 21 hand landmarks per hand.
- ⚡ **Advanced Effects Engine**:
    - Recursive fractal lightning arcs and linear dodge blending.
    - Atmospheric heat haze and multi-radius Gaussian bloom.
- 🖐 **Fist-to-Palm Control**: Clench fists to charge, relax palms to unleash a continuous energy burst.
- 🏔 **Parallax Environment**: Multi-layer mountainous background that moves with 3D depth.
- 🪨 **Physics-Based Debris**: Rocks and dust that "lift off" the ground as your power levels rise.
- 🫨 **Dynamic Camera Tremors**: Screen-shake intensity that scales with your energy level.

---

## 🏗 System Architecture

```
Camera Input
↓
Frame Processor (Face Mesh + Hands)
↓
Gesture Engine (Fist-to-Palm + Proximity)
↓
Effects Engine (Particles + Lightning + Bloom + Shake)
↓
Background Engine (Segmentation + Parallax + Video)
↓
Final Composite
```

---

## 🛠 Technology Stack

- Python 3.10+
- OpenCV
- MediaPipe
- NumPy

---

## 📂 Project Structure (Core)

```
project-saiyan-ar/
│
├── main.py              # Main execution loop and orchestration
├── camera.py            # Webcam abstraction
├── face_tracker.py      # MediaPipe Face Mesh module
├── hand_tracker.py      # MediaPipe Hands module
├── gesture_engine.py    # Gesture state machine (Swipe/Fist/Palm)
├── effects_engine.py    # Cinematic effects (Energy/Rocks/Shake)
├── background_engine.py # Segmentation & Parallax logic
├── utils.py             # Math and coordinate utilities
├── requirements.txt     # Dependency list
└── assets/              # Texture and video assets
```

---

## 🧠 Advanced Gesture Controls

### 1️⃣ Super Saiyan Transformation
**Action**: Swipe your hand horizontally across your face.  
**Effect**: Toggles the Transformation mode. When enabled, your silhouette will glow with electrical arcs!

### 2️⃣ Energy Charge (The Load)
**Action**: Bring both hands together and **clench your fists**.  
**Effect**: A golden energy ball pulses between your hands, rocks start lifting off the ground, and the screen begins to shake.

### 3️⃣ Continuous Burst (Unleash)
**Action**: While hands are together, **relax your palms**.  
**Effect**: Fires a massive, continuous Kamehameha energy blast! Close your fists again to stop the blast.

---

## 🖥 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Mohid-Abbas/RealTime-Gesture-Controlled-AR-Engine.git
cd RealTime-Gesture-Controlled-AR-Engine
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

---

## ‍💻 Author

**Muhammad Mohid Abbas**  
Computer Vision & AI Enthusiast  

⭐ If You Like This Project, give it a star!
