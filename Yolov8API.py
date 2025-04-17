from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
from io import BytesIO
from ultralytics import YOLO
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For all domains; change this to your specific domain for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
model = YOLO('best2.pt')  # Replace with your model path

# Load labels from labels.txt
with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image
        img_bytes = await file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Run YOLO inference
        results = model(img)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        # Prepare the response (coordinates, label, and confidence)
        detected_objects = []
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            label = labels[int(class_id)]
            detected_objects.append({
                "label": label,
                "confidence": float(confidence),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                }
            })

            # Draw boxes (optional, for visual debugging)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert image to JPEG
        _, encoded_img = cv2.imencode('.jpg', img)
        if encoded_img is None:
            raise HTTPException(status_code=500, detail="Error encoding image")

        # Return the image with detected objects and also the list of detected objects in JSON
        return {"image": StreamingResponse(BytesIO(encoded_img.tobytes()), media_type="image/jpeg"),
                "detected_objects": detected_objects}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
