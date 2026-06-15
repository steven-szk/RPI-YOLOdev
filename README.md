# RPI-YOLOdev
Development YOLO for Unibots proj

# New venv with picamera2 system packages
python3 -m venv --system-site-packages .venv

# activate venv 
source .venv/bin/activate

pip install -r requirements.txt

# deactivate venv
deactivate

# IP server:
http://192.168.0.40:1234/

# create ncnn
yolo export model=custom_model_5.pt format=ncnn imgsz=640
