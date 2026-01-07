"""
Retrain models with real face images from UTKFace dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import cv2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add src to path
sys.path.insert(0, '.')

from src.data import TabularPreprocessor, get_qml_features
from src.features.face_cnn import FaceCNN
from src.models import ClassicalTabularModels

print('=' * 60)
print('RETRAINING WITH REAL IMAGES')
print('=' * 60)

# Load real face images
images_dir = Path('data/images/real_faces')
image_files = sorted(list(images_dir.glob('*.jpg')))
print(f'Found {len(image_files)} real face images')

# Load and preprocess images
def load_image(path, size=(224, 224)):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img

images = []
ages = []
for img_path in image_files:
    img = load_image(img_path)
    images.append(img)
    # Extract age from filename
    age_str = img_path.stem.split('_age')[1]
    age = int(age_str.split('.')[0])
    ages.append(age)

images = np.array(images)
ages = np.array(ages)
print(f'Loaded {len(images)} images, ages range: {ages.min()}-{ages.max()}')

# Generate tabular data to match images
n_samples = len(images)
n_hgps = n_samples // 5  # 20% HGPS cases
n_controls = n_samples - n_hgps

print(f'Creating {n_hgps} HGPS and {n_controls} control samples')

# Generate clinical data
np.random.seed(42)
data = []

for i in range(n_samples):
    is_hgps = i < n_hgps
    age = ages[i]

    if is_hgps:
        # HGPS characteristics
        height = 50 + age * 4.5 + np.random.randn() * 3
        weight = 3 + age * 1.2 + np.random.randn() * 1
        small_jaw = int(np.random.random() < 0.85)
        prominent_eyes = int(np.random.random() < 0.80)
        thin_skin = int(np.random.random() < 0.75)
        hair_loss = int(np.random.random() < 0.70)
        lmna_mut = int(np.random.random() < 0.70)
        risk_label = 1
        progression = np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5])
    else:
        # Control characteristics
        height = 50 + age * 5.5 + np.random.randn() * 5
        weight = 3.5 + age * 2.0 + np.random.randn() * 2
        small_jaw = int(np.random.random() < 0.05)
        prominent_eyes = int(np.random.random() < 0.05)
        thin_skin = int(np.random.random() < 0.05)
        hair_loss = int(np.random.random() < 0.02)
        lmna_mut = int(np.random.random() < 0.03)
        risk_label = 0
        progression = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])

    bmi = weight / ((height/100) ** 2)
    expected_height = 50 + age * 5.5
    expected_weight = 3.5 + age * 2.0
    height_z = (height - expected_height) / 5
    weight_z = (weight - expected_weight) / 2

    data.append({
        'patient_id': f'P{i:04d}',
        'age': age,
        'height_cm': height,
        'weight_kg': weight,
        'bmi': bmi,
        'height_z_score': height_z,
        'weight_z_score': weight_z,
        'small_jaw': small_jaw,
        'prominent_eyes': prominent_eyes,
        'thin_skin': thin_skin,
        'hair_loss': hair_loss,
        'lmna_mut': lmna_mut,
        'risk_label': risk_label,
        'progression_label': progression,
        'image_idx': i
    })

df = pd.DataFrame(data)
print(f'Created dataset with {len(df)} samples')
print(f'HGPS: {(df.risk_label == 1).sum()}, Controls: {(df.risk_label == 0).sum()}')

# Split data
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['risk_label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['risk_label'], random_state=42)

print(f'Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')

# Save updated data splits
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

# Fit preprocessor
preprocessor = TabularPreprocessor()
preprocessor.fit(train_df)
joblib.dump(preprocessor, 'models/preprocessor.joblib')

# Prepare features
X_train = preprocessor.transform(train_df)
X_val = preprocessor.transform(val_df)
X_test = preprocessor.transform(test_df)

y_train = train_df['risk_label'].values
y_val = val_df['risk_label'].values
y_test = test_df['risk_label'].values

print('\n' + '=' * 60)
print('TRAINING CLASSICAL ML MODELS')
print('=' * 60)

# Train classical models
classical_models = ClassicalTabularModels(calibrate=True, random_state=42)
classical_models.fit(X_train, y_train, X_val, y_val)
results = classical_models.evaluate(X_test, y_test)

# Save classical models
joblib.dump(classical_models, 'models/classical_models.joblib')

print('\n' + '=' * 60)
print('TRAINING CNN ON REAL IMAGES')
print('=' * 60)

# Create image dataset
class FaceDataset(Dataset):
    def __init__(self, df, images, transform=None):
        self.df = df.reset_index(drop=True)
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_idx = int(row['image_idx'])
        image = self.images[img_idx].copy()

        # Normalize and convert to tensor
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC -> CHW

        return {
            'image': image,
            'risk_label': torch.tensor(row['risk_label'], dtype=torch.long),
            'progression_label': torch.tensor(row['progression_label'], dtype=torch.long)
        }

# Create datasets
train_dataset = FaceDataset(train_df, images)
val_dataset = FaceDataset(val_df, images)
test_dataset = FaceDataset(test_df, images)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Initialize CNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

cnn = FaceCNN(embedding_dim=256, pretrained=True, freeze_backbone=False)
cnn = cnn.to(device)

# Training setup
optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Train CNN
num_epochs = 20
best_val_acc = 0

for epoch in range(num_epochs):
    # Training
    cnn.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch in train_loader:
        images_batch = batch['image'].to(device)
        labels = batch['risk_label'].to(device)

        optimizer.zero_grad()
        outputs = cnn(images_batch)
        loss = criterion(outputs['risk_logits'], labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs['risk_logits'].max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

    # Validation
    cnn.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            images_batch = batch['image'].to(device)
            labels = batch['risk_label'].to(device)

            outputs = cnn(images_batch)
            _, predicted = outputs['risk_logits'].max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    train_acc = 100. * train_correct / train_total
    val_acc = 100. * val_correct / val_total

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'Epoch {epoch+1}/{num_epochs}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(cnn.state_dict(), 'models/face_cnn.pt')

print(f'Best validation accuracy: {best_val_acc:.1f}%')

# Test evaluation
cnn.load_state_dict(torch.load('models/face_cnn.pt', weights_only=True))
cnn.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for batch in test_loader:
        images_batch = batch['image'].to(device)
        labels = batch['risk_label'].to(device)

        outputs = cnn(images_batch)
        _, predicted = outputs['risk_logits'].max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_acc = 100. * test_correct / test_total
print(f'CNN Test accuracy: {test_acc:.1f}%')

print('\n' + '=' * 60)
print('TRAINING QML MODELS')
print('=' * 60)

from src.qml import QuantumSVM, QuantumNeuralNetwork

# Get QML features
X_train_qml, y_train_qml, _ = get_qml_features(train_df, preprocessor)
X_val_qml, y_val_qml, _ = get_qml_features(val_df, preprocessor)
X_test_qml, y_test_qml, _ = get_qml_features(test_df, preprocessor)

# Train QSVM
print('Training QSVM...')
qsvm = QuantumSVM(num_features=6, feature_map_reps=2)
qsvm.fit(X_train_qml, y_train_qml)
qsvm_pred = qsvm.predict(X_test_qml)
qsvm_acc = (qsvm_pred == y_test_qml).mean() * 100
print(f'QSVM Test Accuracy: {qsvm_acc:.1f}%')
joblib.dump(qsvm, 'models/qsvm.joblib')

# Train QNN
print('Training QNN...')
qnn = QuantumNeuralNetwork(num_features=6, ansatz_reps=2)
qnn.fit(X_train_qml, y_train_qml)
qnn_pred = qnn.predict(X_test_qml)
qnn_acc = (qnn_pred == y_test_qml).mean() * 100
print(f'QNN Test Accuracy: {qnn_acc:.1f}%')
joblib.dump(qnn, 'models/qnn.joblib')

# Save training report
import json
from datetime import datetime

report = {
    'timestamp': datetime.now().isoformat(),
    'dataset': {
        'total_images': len(images),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'hgps_samples': int((df.risk_label == 1).sum()),
        'control_samples': int((df.risk_label == 0).sum())
    },
    'results': {
        'classical_ml': results,
        'cnn_test_accuracy': test_acc,
        'qsvm_test_accuracy': qsvm_acc,
        'qnn_test_accuracy': qnn_acc
    }
}

with open('results/training_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

print('\n' + '=' * 60)
print('TRAINING COMPLETE')
print('=' * 60)
print(f'Dataset: {len(images)} real face images')
print(f'CNN: {test_acc:.1f}% test accuracy')
print(f'QSVM: {qsvm_acc:.1f}% test accuracy')
print(f'QNN: {qnn_acc:.1f}% test accuracy')
print('\nAll models saved to models/ directory')
print('Training report saved to results/training_report.json')
