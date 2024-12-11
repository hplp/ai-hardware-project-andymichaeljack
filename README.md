## Team Name: 
Team JAM

## Team Members:
- Andy Chen
- Michael Sekyi
- Jack Hebert

## Project Title: 
Using AI Hardware Acceleration to Detect Real-Time Crowd and Traffic Conditions
## Key Objective:
- Crowd Density Detection: Develop a robust computer vision model using Hailo 8 on Raspberry Pi to detect crowd density in real-time and categorize it into different levels (e.g., low, medium, high) based on traffic patterns along Engineer's Way
- Energy Efficiency and Processing Optimization: Leverage the hardware acceleration provided by Hailo 8 to ensure the system runs efficiently, with low latency, minimal energy consumption, and optimized processing for real-time applications
- Scalability and Deployment Strategy: Design a scalable solution that can easily be deployed across multiple pathways or locations on campus, providing consistent and accurate data on foot traffic

## Technology Stack
- Hardware: Raspberry Pi, Hailo 8 AI processor, Pixy2 camera
- Software: Python, OpenCV, TensorFlow Lite, Hailo SDK

## Expected Outcomes:
- Real-Time Traffic Analysis Model: A working computer vision model that identifies and categorizes crowd density levels on Engineer's Way with at least 70% accuracy
- Low-Power Edge AI Solution: A model optimized to run on the Raspberry Pi + Hailo 8 with minimal power consumption while maintaining real-time performance
- Scalable Documentation: Detailed project documentation, including setup guides, configuration steps, and recommendations for deploying additional nodes on other pathways

## Timeline
Week 1: Planning and Research

Week 2: Data Collection and Model Training

Week 3: Model Optimization and Testing

Week 4: System Integration and Deployment

---

### Software Requirements
- **Python**: Version 3.7 or later recommended
- Python Libraries:
  - TensorFlow 1.x (using `tensorflow.compat.v1`)
  - OpenCV (`opencv-python`)
  - ProgressBar2 (`progressbar2`)
  - NumPy (`numpy`)

### 1. Clone the Repository
Download the repository to your local machine:
```bash
git clone <your-repo-url>
cd <repository-folder>
```

### 2. Install Dependencies
Install all required Python packages:
```bash
pip install tensorflow==1.15 opencv-python progressbar2 numpy
```

### 3. Prepare the Model
Place your TensorFlow frozen graph file (`my_model.pb`) in the following path:
```
./data/utils/my_model.pb
```

### 4. Prepare Test Images
Place your test images (JPG format) in the following directory:
```
./data/images/test/
```

---

## **Running the Script**

### 1. Open a Terminal
Navigate to the repository folder

### 2. Run the Script
Execute the following command:
```bash
python people_counter.py
```

---

## **Results**
- Processed images will be saved in the `./results/` folder
- Each image will display:
  - Bounding boxes around detected people
  - The count of detected people
  - The crowd level (Low, Medium, High)
 
- **Our Results**
  - [Image 1](CrowdDetection/results/IMG_1650.jpg)
  - [Image 2](CrowdDetection/results/crowd-inside-shopping-mall-protest-gathering-CBP35C.jpg)
  - [Image 3](CrowdDetection/results/aerial-view-city-people-8ba3646909cce4de5bd4249134d24d63.jpg)
- Further results can be found in the `./results/` folder

---

## **Troubleshooting**

### Missing Model File Error
Ensure the `my_model.pb` file is in the correct location (`./data/utils/`)

### No Test Images Found
Verify that the `./data/images/test/` directory contains `.jpg` images

### Missing Results Folder
If the script fails to save results, manually create the `./results/` folder






[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Buol6fpg)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=16865888)
