# import cv2
# import numpy as np
from pathlib import Path
import json

data_dir = Path("data/faces")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
print("bye")
faces = []
labels = []

label_map = {}
current_label = 0

# Read each person's folder
for person_dir in sorted(data_dir.iterdir()):

    if not person_dir.is_dir():
        continue

    person_name = person_dir.name

    label_map[current_label] = person_name

    print(f"Loading images for {person_name}...")

    for image_path in person_dir.glob("*.jpg"):

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        image = cv2.resize(image, (200, 200))

        faces.append(image)
        labels.append(current_label)

    current_label += 1


if len(faces) == 0:
    print("No training images found.")
    exit()


# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train
recognizer.train(
    faces,
    np.array(labels)
)

# Save trained model
recognizer.write("models/face_model.yml")


# Save label mapping
with open("models/labels.json", "w") as file:
    json.dump(label_map, file)


print()
print("Training completed!")
print(f"Total images used: {len(faces)}")
print(f"People enrolled: {len(label_map)}")
print("Model saved to models/face_model.yml")