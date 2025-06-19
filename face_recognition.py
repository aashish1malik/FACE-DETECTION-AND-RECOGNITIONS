import cv2
import numpy as np
from ultralytics import YOLO
from keras_facenet import FaceNet
import faiss
from sklearn.svm import SVC
from sklearn.exceptions import NotFittedError
from sklearn.calibration import CalibratedClassifierCV


face_detector =  YOLO('yolov8n.pt')   
embedder = FaceNet()  


index = faiss.IndexFlatL2(512)  


face_db = {
    "embeddings": [],
    "labels": [],
    "mask_probs": [],
    "cap_probs": []
}


mask_classifier = CalibratedClassifierCV(
    SVC(kernel='rbf', C=10, probability=True)
)
cap_classifier = CalibratedClassifierCV(
    SVC(kernel='rbf', C=10, probability=True)
)

def align_face(image, landmarks):
    """Align face using 5-point landmarks"""
    
    eye_center = ((landmarks[0][0] + landmarks[1][0]) / 2,
                  (landmarks[0][1] + landmarks[1][1]) / 2)
    
    dy = landmarks[1][1] - landmarks[0][1]
    dx = landmarks[1][0] - landmarks[0][0]
    angle = np.degrees(np.arctan2(dy, dx))
    
    
    M = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                            flags=cv2.INTER_CUBIC)
    
    return aligned

def get_embedding(face_image):
    """Get normalized FaceNet embedding"""
  
    face = cv2.resize(face_image, (160, 160))
    face = (face.astype('float32') - 127.5) / 128.0  
    return embedder.embeddings(np.expand_dims(face, axis=0))[0]

def detect_accessories(embedding):
    try:
        mask_prob = mask_classifier.predict_proba([embedding])[0][1]
        cap_prob = cap_classifier.predict_proba([embedding])[0][1]
        return mask_prob > 0.98, cap_prob > 0.98
    except NotFittedError:
        
        return False, False

def register_person(image_path, label):
    """Register a new face in the database"""
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    
    results = face_detector(image)
    if len(results[0]) == 0:
        raise ValueError("No faces detected")
    
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    main_face_idx = np.argmax(areas)
    x1, y1, x2, y2 = map(int, boxes[main_face_idx])
    
   
    face = image[y1:y2, x1:x2]
    landmarks = [
        (x1 + (x2-x1)*0.3, y1 + (y2-y1)*0.35),  
        (x1 + (x2-x1)*0.7, y1 + (y2-y1)*0.35), 
        (x1 + (x2-x1)*0.5, y1 + (y2-y1)*0.6),   
        (x1 + (x2-x1)*0.3, y1 + (y2-y1)*0.8),  
        (x1 + (x2-x1)*0.7, y1 + (y2-y1)*0.8)   
    ]
    
   
    aligned = align_face(image, landmarks)
    aligned_face = aligned[y1:y2, x1:x2]
    embedding = get_embedding(aligned_face)
    
    
    mask_prob, cap_prob = detect_accessories(embedding)
    
   
    face_db["embeddings"].append(embedding)
    face_db["labels"].append(label)
    face_db["mask_probs"].append(mask_prob)
    face_db["cap_probs"].append(cap_prob)
    index.add(np.array([embedding], dtype='float32'))
    
    return f"Registered {label} (Mask: {mask_prob:.1%}, Cap: {cap_prob:.1%})"

def recognize_faces(image_path, threshold=0.98):
    """Recognize faces with high confidence threshold"""
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    results = face_detector(image)
    
    recognitions = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        face = image[y1:y2, x1:x2]
        
        
        face = cv2.resize(face, (160, 160))
        embedding = get_embedding(face)
        
        
        D, I = index.search(np.array([embedding], dtype='float32'), k=1)
        distance = D[0][0]
        
        if distance > 1.0:  
            recognitions.append({
                "label": "Unknown",
                "confidence": 0,
                "box": [x1, y1, x2, y2]
            })
            continue
            
        idx = I[0][0]
        confidence = 1 - distance  
        
        if confidence >= threshold:  
            mask_prob, cap_prob = detect_accessories(embedding)
            recognitions.append({
                "label": face_db["labels"][idx],
                "confidence": confidence,
                "mask": mask_prob >= 0.98,
                "cap": cap_prob >= 0.98,
                "box": [x1, y1, x2, y2]
            })
        else:
            recognitions.append({
                "label": "Unknown",
                "confidence": confidence,
                "box": [x1, y1, x2, y2]
            })
    
    return recognitions

def train_accessory_detectors(mask_samples, cap_samples):
    """Train high-accuracy accessory classifiers"""
    X = []
    y_mask = []
    y_cap = []
    
    for sample in mask_samples:
        embedding = get_embedding(sample["image"])
        X.append(embedding)
        y_mask.append(1 if sample["has_mask"] else 0)
    
    for sample in cap_samples:
        embedding = get_embedding(sample["image"])
        X.append(embedding)
        y_cap.append(1 if sample["has_cap"] else 0)
    
    mask_classifier.fit(X, y_mask)
    cap_classifier.fit(X, y_cap)
    
    
    mask_acc = mask_classifier.score(X, y_mask)
    cap_acc = cap_classifier.score(X, y_cap)
    print(f"Mask detector accuracy: {mask_acc:.2%}")
    print(f"Cap detector accuracy: {cap_acc:.2%}")


if __name__ == "__main__":
   
    print(register_person("image_2.jpg", "John"))
    print(register_person("image_1.jpg", "Sarah"))
    
   
    results = recognize_faces("group_project1.jpg")
    for r in results:
        print(f"{r['label']} ({r['confidence']:.1%})")
        print(f"Accessories: Mask={r.get('mask', False)}, Cap={r.get('cap', False)}")
        
        
