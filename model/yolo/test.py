from ultralytics import YOLO

# Load model
model_path = r"C:\Users\ASUS\Documents\RS UGM\best.pt"

# Load Ultralytics YOLO model
model = YOLO(model_path)

# Load image
image_path = r"C:\Users\ASUS\Documents\RS UGM\model\yolo\test\images\dr Arso"

# Perform inference
results = model(image_path, save=True, show_labels=True, show_conf=True)
