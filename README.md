# Truck Safety: Forward Collision and Lane Boundary Detection System

## Project Overview
The **Truck Safety project** is a 2024-25 Senior Design Project. Real-time collision warning and lane detection system for commercial vehicles using computer vision and edge AI.

## What it does
- **Forward Collision Warning (FCW)**: Detects vehicles and pedestrians in real-time, warning drivers to prevent collisions (30 FPS on test videos).
- **Lane Boundary Detection **: Identifies lane markings using classical vision techniques.

## What I learned
- **Pipeline Design**: Built a modular Python pipeline that processes camera feeds, runs object detection, and performs lane detection using OpenCV computer vision techniques **.
- **Data Processing**: Used BDD100K dataset for training and validation, preprocessing vehicle and pedestrian detection data for optimal model performance.
- **Edge Optimization**: Optimized the system for real-time performance on Raspberry Pi 5 with Hailo 8L, achieving 30 FPS processing speeds with low latency.


## Setup Instructions
To run the Truck Safety system, follow these steps to configure the environment and install dependencies.

### 1. Environmental Configuration
Activate the Hailo virtual environment using the provided script:
```bash
source setup_env.sh
```
If version conflicts occur, install specific Hailo packages:
```bash
sudo apt install hailo-tappas-core=3.30.0-1 hailo-dkms=4.19.0-1 hailort=4.19.0-3
```

### 2. Requirement Installation
In the activated virtual environment, install Python dependencies:
```bash
pip install -r requirements.txt
```

### 3. Resource Download
Download required resources (e.g., YOLOv10s HEF model):
```bash
./download_resources.sh
```

## Running the Code
Run the main pipeline with:
```bash
python basic_pipelines/TPD_detection.py
```

### Options
- `--input rpi`: Use RPi camera input for real-time detection.
- `--input /path/to/file`: Specify a video/image file (e.g., `test01.mp4`).
- `--show-fps`: Display frames per second on the screen.
- `--hef`: Path to the Hailo HEF model (e.g., `yolov10s.hef`).

### Example
```bash
python basic_pipelines/TPD_detection.py --hef resources/yolov10s.hef --input resources/test01.mp4 --show-fps
```

## Future Work
- **HAWQ Quantization**: Implement Hessian-based sensitivity analysis in C++ to further compress YOLOv10s, targeting <1% mAP loss.
- **Scalability**: Extend the pipeline to support multi-camera inputs and cloud integration, leveraging Java/C# for backend services.

## Contact and Portfolio
I’m Jeongmoo Yoo, a UC Irvine CSE graduate passionate about AV and embedded systems. Connect with me to discuss this project or software engineering opportunities:
- **GitHub**: [github.com/jungyoo311](https://github.com/jungyoo311)
- **LinkedIn**: [linkedin.com/in/jungyoo-cse](https://linkedin.com/in/yourprofile)
- **Email**: [jeongmy@uci.edu](mailto:jeongmy@uci.edu)

Explore my other AV projects, including **YOLO-NAS** training on **BDD100K** and **HAWQ quantization**, to see my work in optimizing edge AI for safety-critical applications.
