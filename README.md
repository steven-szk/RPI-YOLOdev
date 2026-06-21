# RPI-YOLOdev
Development YOLO for Unibots proj

ssh steven@192.168.0.40

# New venv with picamera2 system packages
python3 -m venv --system-site-packages .venv

# activate venv 
source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# deactivate venv
deactivate

# IP server:
http://192.168.0.40:1234/

# create ncnn
yolo export model=custom_model_5.pt format=ncnn imgsz=640


# NEW PI COMMAND:
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
sudo apt install -y python3-picamera2 python3-libcamera
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
yolo export model=custom_model_5.pt format=ncnn imgsz=640

