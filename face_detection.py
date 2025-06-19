from ultralytics import YOLO
import cv2
import os


model = YOLO('yolov8n.pt') 

def detect_faces(image_path):
    """Detect faces in an image with error handling"""
   
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
   
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"OpenCV could not read image at: {image_path}")
    
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    results = model(image_rgb)
    
   
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            face = image[y1:y2, x1:x2]
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': confidence,
                'face_image': face
            })
    
    return detections

if __name__ == "__main__":
    
    test_image = os.path.abspath("test.png")
    
    try:
        detections = detect_faces(test_image)
        print(f"Found {len(detections)} faces:")
        
        for i, detection in enumerate(detections, 1):
            print(f"Face {i}:")
            print(f"  Position: {detection['bbox']}")
            print(f"  Confidence: {detection['confidence']:.2%}")
            
            
            output_path = f"face_{i}.jpg"
            cv2.imwrite(output_path, detection['face_image'])
            print(f"  Saved as {output_path}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Possible solutions:")
        print("1. Ensure 'test.jpg' exists in your project folder")
        print("2. Try using an absolute path to the image")
        print("3. Verify the image is not corrupted")