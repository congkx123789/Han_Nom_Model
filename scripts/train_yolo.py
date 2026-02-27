from ultralytics import YOLO
import os

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data='data/yolo_dataset/data.yaml',  # path to data.yaml
    epochs=50,  # number of epochs
    imgsz=640,  # training image size
    batch=16,   # batch size
    name='yolov8n_hannom', # name of the experiment
    project='runs/detect', # project name
    device='0', # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    patience=10, # early stopping
    save=True, # save checkpoint
)

print("Training complete.")
print(f"Best model saved at: {results.save_dir}/weights/best.pt")
