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

# Configuration
DATA_DIR = 'raw_dataset'  # Path to your dataset
MODEL_PATH = 'face_recognition_model.pkl'
RECOGNITION_THRESHOLD = 0.6  # Confidence threshold (adjust as needed)
CLASS_NAMES = ["Arslan", "Praful", "Sanket", "Rizwan", "Swastik"]  # Class-ID mapping

# Initialize FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
transform = transforms.Compose([
    transforms.Resize(160),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# YOLO annotation parser (unchanged)
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

        # Expand face area by 20% for better recognition
        expand_w = int((x2 - x1) * 0.2)
        expand_h = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - expand_w)
        y1 = max(0, y1 - expand_h)
        x2 = min(w, x2 + expand_w)
        y2 = min(h, y2 + expand_h)

        face = img[y1:y2, x1:x2]
        if face.size > 0:
            faces.append((face, int(class_id)))

    return faces

# Corrected data preparation function
def prepare_training_data():
    features = []
    labels = []
    class_names = CLASS_NAMES  # Use predefined class names

    print(f"\n{'=' * 40}")
    print(f"Starting data preparation")
    print(f"Found {len(class_names)} classes: {class_names}")

    image_dir = os.path.join(DATA_DIR, 'images')
    label_dir = os.path.join(DATA_DIR, 'labels')

    print(f"\nScanning images in: {image_dir}")
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} image files")

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(image_dir, img_file)
        txt_path = os.path.join(label_dir, base_name + '.txt')

        print(f"\nProcessing: {img_file}")
        print(f"Label path: {txt_path}")

        if not os.path.exists(txt_path):
            print("❌ Missing label file!")
            continue

        faces = parse_yolo_annotation(img_path, txt_path)
        print(f"Found {len(faces)} faces in this image")

        for i, (face_img, class_id) in enumerate(faces):
            print(f"  Face {i + 1}: Class ID {class_id}", end=' ')
            if class_id >= len(class_names):
                print(f"(Invalid: max class is {len(class_names) - 1})")
                continue

            name = class_names[class_id]
            embedding = get_embedding(face_img)
            if embedding is not None:
                features.append(embedding)
                labels.append(name)
                print(f"- Valid ({name})")
            else:
                print("- Failed embedding")

    print(f"\n{'=' * 40}")
    print(f"Total training samples: {len(features)}")
    print(f"Class distribution:")
    for name in class_names:
        count = labels.count(name)
        print(f"  {name}: {count} samples")

    return np.array(features), np.array(labels)

# Feature extraction (unchanged)
def get_embedding(face_img):
    try:
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet(face_tensor)
        return embedding.cpu().numpy().flatten()
    except:
        return None

# Train model (unchanged)
def train_model():
    features, labels = prepare_training_data()

    if len(features) == 0:
        raise ValueError("No training data found!")

    le = LabelEncoder()
    y = le.fit_transform(labels)

    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=3, weights='distance')
    )
    model.fit(features, y)

    joblib.dump((model, le), MODEL_PATH)
    print(f"Trained on {len(features)} samples. Classes: {le.classes_}")
    return model, le

# Real-time recognition (unchanged)
def real_time_recognition():
    try:
        model, le = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Model not found! Training first...")
        model, le = train_model()

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            face_img = rgb[y:y + h, x:x + w]
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

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{pred_class} ({max_prob:.2f})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_model()
    real_time_recognition()