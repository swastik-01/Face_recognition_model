import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image
from ultralytics import YOLO

# Configuration
DATA_DIR = 'raw_dataset'
MODEL_PATH = 'face_recognition_model.pkl'
RECOGNITION_THRESHOLD = 0.6
CLASS_NAMES = ["Arslan", "Praful", "Sanket", "Rizwan", "Swastik"]
YOLO_MODEL_PATH = 'yolov8n-face.pt'  # Use custom trained face detection model

# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
face_detector = YOLO(YOLO_MODEL_PATH).to(device)
transform = transforms.Compose([
    transforms.Resize(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# Optimized YOLO annotation parser
def parse_yolo_annotation(img_path, txt_path):
    img = cv2.imread(img_path)
    if img is None:
        return []

    h, w = img.shape[:2]
    faces = []

    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, x_center, y_center, width, height = map(float, parts)
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        # Adaptive expansion based on face size
        expand_factor = 0.25 if (x2 - x1) < 100 else 0.15
        expand_w = int((x2 - x1) * expand_factor)
        expand_h = int((y2 - y1) * expand_factor)

        x1 = max(0, x1 - expand_w)
        y1 = max(0, y1 - expand_h)
        x2 = min(w, x2 + expand_w)
        y2 = min(h, y2 + expand_h)

        face = img[y1:y2, x1:x2]
        if face.size > 0:
            faces.append((face, int(class_id)))

    return faces


# Optimized feature extraction with batch processing
def get_embedding(face_img):
    try:
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            embedding = resnet(face_tensor)
        return embedding.cpu().numpy().flatten()
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


# Enhanced real-time recognition with YOLOv8
def real_time_recognition():
    try:
        model, le = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Model not found! Training first...")
        model, le = train_model()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 face detection with GPU acceleration
        results = face_detector.predict(frame, imgsz=640, conf=0.7, device=device)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), conf in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                face_img = frame[y1:y2, x1:x2]

                embedding = get_embedding(face_img)
                if embedding is None:
                    continue

                proba = model.predict_proba([embedding])[0]
                max_prob = np.max(proba)

                if max_prob > RECOGNITION_THRESHOLD:
                    pred_class = le.inverse_transform([np.argmax(proba)])[0]
                    color = (0, 255, 0)
                else:
                    pred_class = "Unknown"
                    color = (0, 0, 255)

                # Enhanced visualization
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{pred_class} ",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Det Conf: {conf:.2f}",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Prepares data and returns embeddings and labels
def prepare_training_data():
    features = []
    labels = []

    image_dir = os.path.join(DATA_DIR, 'images')
    label_dir = os.path.join(DATA_DIR, 'labels')

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(image_dir, img_file)
        txt_path = os.path.join(label_dir, base_name + '.txt')

        if not os.path.exists(txt_path):
            continue

        faces = parse_yolo_annotation(img_path, txt_path)
        for i, (face_img, class_id) in enumerate(faces):
            if class_id >= len(CLASS_NAMES):
                continue

            name = CLASS_NAMES[class_id]
            embedding = get_embedding(face_img)
            if embedding is not None:
                features.append(embedding)
                labels.append(name)

    return np.array(features), np.array(labels)



# Trains the model and saves it
def train_model():
    embeddings, labels = prepare_training_data()

    if len(embeddings) == 0:
        print("No training data found.")
        return None, None

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3, weights='distance'))
    model.fit(embeddings, labels_encoded)

    joblib.dump((model, le), MODEL_PATH)
    print("Model trained and saved.")

    return model, le


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_model()
    real_time_recognition()