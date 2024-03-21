from ultralytics import YOLO

def train_yolo(model_type, data_path, epochs):
    
    # Choose the appropriate YOLO model based on the user's input
    if model_type == 'yolov8n':
        model = YOLO('/content/drArso/model/yolo/yolov8n.pt')
    elif model_type == 'yolov8s':
        model = YOLO('/content/drArso/model/yolo/yolov8s.pt')
    elif model_type == 'yolov8m':
        model = YOLO('/content/drArso/model/yolo/yolov8m.pt')
    elif model_type == 'yolov8l':
        model = YOLO('/content/drArso/model/yolo/yolov8l.pt')
    elif model_type == 'yolov8x':
        model = YOLO('/content/drArso/model/yolo/yolov8x.pt')
    else:
        raise ValueError("Invalid model type. Supported types: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x")

    results = model.train(
        data   = data_path,
        epochs = epochs
    )

# Example usage
model_type = input("Enter YOLO model type (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x): ")
data_path = '/content/drArso/model/yolo/coco.yaml'
epochs = 100

train_yolo(model_type, data_path, epochs)
