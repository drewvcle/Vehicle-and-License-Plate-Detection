## Vehicle Detection

Utilized YOLOv5 to train model on a custom dataset.

### Setup Instructions

1. Create a virtual environment:

   `py -3.10 -m venv env`

2. Activate the virtual environment:

   `.\env\Scripts\Activate.ps1`

3. Install project dependencies:

   `pip install -r requirements.txt`

### Model Training

4. Fine-tune a pre-trained YOLOv5 model:

   `python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5m.pt`

### Inference

5. Detect vehicles in a single image:

   `python detect.py --weights runs/train/exp12/weights/best.pt --source test_images/your_image.jpg`

6. Detect vehicles in a video (Default video path:`my_test_vids\v1.mp4`):  
   
   `python tracker.py`

7. Real-time vehicle detection using device’s webcam:

   `python tracker_webcam.py`

## License Plate Detection

Utilized pre-trained model (YOLOv8n) to detect vehicles, and YOLOv8 to detect license plates.

### Setup Instructions

1. Create a virtual environment:  

   `py -3.10 -m venv env`

2. Activate the virtual environment:  

   `.\env\Scripts\Activate.ps1`

3. Install project dependencies:  

   `pip install -r requirements.txt`

### Pipeline

4. Run main.py. Ensure video sample is named accordingly (default is 'sample.mp4'):
   Ensure the input video is named correctly (default: `sample.mp4`)  

   `python main.py`

5. Run the add_missing_data.py to interpolate values for missing frames: 

   `python interpolate.py`

6. Draw borders and cropped license plate into video (default output is out.mp4):

   `python finalize.py`

