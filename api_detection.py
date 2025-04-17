from fastapi import FastAPI, File, UploadFile
import torch
import cv2
import numpy as np
from pydantic import BaseModel
import io

app = FastAPI()

# Load YOLOv5 model (use the correct method based on how the model was saved)
model = torch.load('best2.pt', weights_only=True)  # Local file loading
model.eval()  # Set the model to evaluation mode

# Helper function to process and detect objects
def detect_objects(image_bytes: bytes):
    # Convert bytes to numpy array
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Resize the image to the input size of YOLOv5 model (640x640)
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Convert the image to a tensor
    img_tensor = torch.from_numpy(img_rgb).float()
    img_tensor /= 255.0  # Normalize the image
    img_tensor = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)  # Convert to NCHW format

    # Perform inference
    with torch.no_grad():
        predictions = model(img_tensor)

    # Extract the predictions (class ids, boxes, and confidence scores)
    pred = predictions[0]  # Get the first prediction
    pred = pred[pred[:, 4] > 0.5]  # Filter by confidence score

    # Format the results
    class_ids = pred[:, 5].tolist()
    boxes = pred[:, :4].tolist()  # x1, y1, x2, y2 bounding boxes
    confidences = pred[:, 4].tolist()

    return class_ids, boxes, confidences

class PredictionResult(BaseModel):
    class_ids: list
    boxes: list
    confidences: list

@app.post("/predict/", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    class_ids, boxes, confidences = detect_objects(image_bytes)
    return PredictionResult(class_ids=class_ids, boxes=boxes, confidences=confidences)
