Real-time object detection application using Python and the YOLOv8 model 
# 🚗 Adaptive Headlights Control for Vehicles

## 📌 Project Overview
Nighttime driving visibility and the temporary blinding caused by oncoming headlights are major factors in traffic accidents.This project implements an intelligent Advanced Driver Assistance System (ADAS) that automates the switching between High Beam and Low Beam, eliminating human error and increasing driving comfort. 

Using a custom-trained **YOLOv8** model, the software processes real-time video streams to detect oncoming traffic and dynamically adjusts the lighting system based on distance estimation.

## 🚀 Key Features
**Real-Time Night Traffic Detection**: Identifies vehicles (cars, trucks), headlights, and taillights in low-visibility conditions.
***Distance Estimation Logic**: Calculates relative distance inversely proportional to the bounding box dimensions. 
***Automated Beam Switching**: If an oncoming vehicle's apparent size exceeds the safety threshold of 350 pixels, the system immediately switches to **Low Beam** to prevent blinding.Once the road is clear, it reverts to **High Beam**.
***Video & Static Simulation**: Fully validated on independent test images and controlled video scenarios.

## 🧠 Model Architecture & Training
The core of this project is powered by **YOLOv8** (You Only Look Once), chosen for its superior real-time processing speed and anchor-free architecture.
***Architecture**: Utilizes a modified CSPDarknet53 Backbone, a PANet Neck for feature aggregation, and a decoupled Head for precise bounding box regression.
**Dataset Prep**: The dataset contains diverse night traffic scenarios and was preprocessed, augmented, and annotated using **Roboflow**.
**Fine-Tuning**: Trained via Transfer Learning for 30-50 epochs using a batch size of 16. 
**Performance**: Achieved high Mean Average Precision (mAP) and a balanced F1-Score maximized at a confidence threshold of 0.183 to filter out background noise.

## 📂 Repository Structure
* `main.py` / `Model testing.py`: Core inference and beam-switching logic scripts.
* `analiza_si_antrenare.ipynb`: Jupyter notebook detailing the data analysis and training process.
* `requirements.txt`: Python dependencies required to run the project.
* `Demo_Rezultate/`: Visual proofs of the model detecting vehicles and switching beams.
* `DOCUMENTATIE_SBC_ILBAN_ALESSANDRO.pdf`: Full technical documentation (Romanian) covering the dataset distribution, confusion matrix, and algorithmic logic.

## 🛠️ Tech Stack
* **Python** * **Ultralytics (YOLOv8)**
* **Computer Vision** (Object Detection, Bounding Box Math)
