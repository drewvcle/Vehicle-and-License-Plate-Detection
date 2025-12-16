## Vehicle Detection

Utilized YOLOv5 to train model on a custom dataset.

### Setup Instructions

1. Create a virtual environment

   `py -3.10 -m venv env`

3. Activate the virtual environment
   .\env\Scripts\Activate.ps1

4. Install project dependencies
   pip install -r requirements.txt

### Model Training

4. Fine-tune a pre-trained YOLOv5 model
   python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5m.pt

### Inference

5. Detect vehicles in a single image
   python detect.py --weights runs/train/exp12/weights/best.pt --source test_images/your_image.jpg

6. Detect vehicles in a video  
   Default video path: my_test_vids\v1.mp4

   python tracker.py

7. Real-time vehicle detection using device’s webcam
   python tracker_webcam.py

## License Plate Detection

Utilized pre-trained model (YOLOv8n) to detect vehicles, and YOLOv8 to detect license plates.

### Setup Instructions

1. Create a virtual environment  
   py -3.10 -m venv env

2. Activate the virtual environment  
   .\env\Scripts\Activate.ps1

3. Install project dependencies  
   pip install -r requirements.txt

### Pipeline

4. Run the main detection and tracking pipeline  
   Ensure the input video is named correctly (default: `sample.mp4`)  
   python main.py

5. Interpolate missing detections across frames  
   This step fills in bounding boxes for frames where detections were missed  
   python interpolate.py

6. Generate the final annotated video with bounding boxes and cropped license plates  
   Default output file: `out.mp4`  
   python finalize.py

