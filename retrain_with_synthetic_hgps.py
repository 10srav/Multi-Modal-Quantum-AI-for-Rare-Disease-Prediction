"""
Retrain models with synthetic HGPS images + control images.

This script trains the model using:
- Real control faces from UTKFace
- Synthetic HGPS-like faces generated locally

FOR LOCAL DEMO USE ONLY - Do not upload trained models with synthetic HGPS data publicly.
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
print('RETRAINING WITH SYNTHETIC HGPS IMAGES')
print('FOR LOCAL DEMO USE ONLY')
print('=' * 60)

# Paths
CONTROL_DIR = Path('data/images/real_faces')
HGPS_DIR = Path('data/images/synthetic_hgps')

# Load control images
control_files = sorted(list(CONTROL_DIR.glob('*.jpg')))
print(f'Found {len(control_files)} control images')

# Load synthetic HGPS images
hgps_files = sorted(list(HGPS_DIR.glob('*.jpg')))
print(f'Found {len(hgps_files)} synthetic HGPS images')

def load_image(path, size=(224, 224)):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img

def extract_age_from_filename(filename):
    """Extract age from filename like 'child_000_age5.jpg' or 'hgps_mild_000.jpg'"""
    stem = Path(filename).stem
    if '_age' in stem:
        try:
            age_str = stem.split('_age')[1].split('.')[0]
            return int(age_str)
        except:
            pass
    # Default age for HGPS synthetic images
    return np.random.randint(2, 10)

# Load all images
print('\nLoading images...')
images = []
labels = []  # 0 = control, 1 = HGPS
ages = []
sources = []

# Load control images (label = 0)
for img_path in control_files[:35]:  # Use 35 controls
    img = load_image(img_path)
    images.append(img)
    labels.append(0)
    ages.append(extract_age_from_filename(img_path.name))
    sources.append('control')

# Load HGPS synthetic images (label = 1)
for img_path in hgps_files:
    img = load_image(img_path)
    images.append(img)
    labels.append(1)
    ages.append(extract_age_from_filename(img_path.name))
    sources.append('hgps_synthetic')

images = np.array(images)
labels = np.array(labels)
ages = np.array(ages)

print(f'Total samples: {len(images)}')
print(f'Control: {(labels == 0).sum()}, HGPS: {(labels == 1).sum()}')
print(f'Age range: {ages.min()}-{ages.max()} years')

# Generate clinical data to match
np.random.seed(42)
data = []

for i in range(len(images)):
    is_hgps = labels[i] == 1
    age = ages[i]

    if is_hgps:
        # HGPS characteristics - more pronounced
        height = 50 + age * 4.2 + np.random.randn() * 2.5  # Shorter
        weight = 3 + age * 1.0 + np.random.randn() * 0.8   # Lighter
        small_jaw = int(np.random.random() < 0.90)         # Very common
        prominent_eyes = int(np.random.random() < 0.85)
        thin_skin = int(np.random.random() < 0.85)
        hair_loss = int(np.random.random() < 0.80)
        lmna_mut = int(np.random.random() < 0.75)
        risk_label = 1
        # Progression based on severity in filename
        if 'severe' in sources[i] or (sources[i] == 'hgps_synthetic' and i % 3 == 2):
            progression = 2  # Rapid
        elif 'moderate' in sources[i] or (sources[i] == 'hgps_synthetic' and i % 3 == 1):
            progression = 1  # Moderate
        else:
            progression = 0  # Slow
    else:
        # Control characteristics
        height = 50 + age * 5.5 + np.random.randn() * 5
        weight = 3.5 + age * 2.0 + np.random.randn() * 2
        small_jaw = int(np.random.random() < 0.03)
        prominent_eyes = int(np.random.random() < 0.03)
        thin_skin = int(np.random.random() < 0.03)
        hair_loss = int(np.random.random() < 0.01)
        lmna_mut = int(np.random.random() < 0.01)
        risk_label = 0
        progression = 0

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
        'image_idx': i,
        'source': sources[i]
    })

df = pd.DataFrame(data)
print(f'\nDataset created: {len(df)} samples')
print(f'HGPS: {(df.risk_label == 1).sum()}, Controls: {(df.risk_label == 0).sum()}')

# Split data (stratified)
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['risk_label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['risk_label'], random_state=42)

print(f'Train: {len(train_df)} (HGPS: {(train_df.risk_label==1).sum()})')
print(f'Val: {len(val_df)} (HGPS: {(val_df.risk_label==1).sum()})')
print(f'Test: {len(test_df)} (HGPS: {(test_df.risk_label==1).sum()})')

# Save data splits (locally, not for upload)
train_df.to_csv('data/train_synthetic.csv', index=False)
val_df.to_csv('data/val_synthetic.csv', index=False)
test_df.to_csv('data/test_synthetic.csv', index=False)

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

for model_name, metrics in results.items():
    print(f"{model_name}: Acc={metrics['accuracy']:.2%}, F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")

# Save classical models
joblib.dump(classical_models, 'models/classical_models.joblib')

print('\n' + '=' * 60)
print('TRAINING CNN ON SYNTHETIC HGPS + CONTROL IMAGES')
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

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Initialize CNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

cnn = FaceCNN(embedding_dim=256, pretrained=True, freeze_backbone=False)
cnn = cnn.to(device)

# Training setup with class weights for imbalanced data
class_counts = np.bincount(y_train)
class_weights = torch.FloatTensor([1.0 / c for c in class_counts]).to(device)
class_weights = class_weights / class_weights.sum() * 2

optimizer = torch.optim.Adam(cnn.parameters(), lr=5e-5, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Train CNN
num_epochs = 30
best_val_acc = 0

for epoch in range(num_epochs):
    # Training
    cnn.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch in train_loader:
        images_batch = batch['image'].to(device)
        labels_batch = batch['risk_label'].to(device)

        optimizer.zero_grad()
        outputs = cnn(images_batch)
        loss = criterion(outputs['risk_logits'], labels_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs['risk_logits'].max(1)
        train_total += labels_batch.size(0)
        train_correct += predicted.eq(labels_batch).sum().item()

    scheduler.step()

    # Validation
    cnn.eval()
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            images_batch = batch['image'].to(device)
            labels_batch = batch['risk_label'].to(device)

            outputs = cnn(images_batch)
            _, predicted = outputs['risk_logits'].max(1)
            val_total += labels_batch.size(0)
            val_correct += predicted.eq(labels_batch).sum().item()
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels_batch.cpu().numpy())

    train_acc = 100. * train_correct / train_total
    val_acc = 100. * val_correct / val_total

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'Epoch {epoch+1}/{num_epochs}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(cnn.state_dict(), 'models/face_cnn.pt')

print(f'\nBest validation accuracy: {best_val_acc:.1f}%')

# Test evaluation
cnn.load_state_dict(torch.load('models/face_cnn.pt', weights_only=True))
cnn.eval()

test_preds = []
test_labels = []

with torch.no_grad():
    for batch in test_loader:
        images_batch = batch['image'].to(device)
        labels_batch = batch['risk_label'].to(device)

        outputs = cnn(images_batch)
        _, predicted = outputs['risk_logits'].max(1)
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(labels_batch.cpu().numpy())

test_acc = accuracy_score(test_labels, test_preds) * 100
test_f1 = f1_score(test_labels, test_preds, average='weighted')
print(f'CNN Test Accuracy: {test_acc:.1f}%')
print(f'CNN Test F1: {test_f1:.3f}')

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
    'note': 'LOCAL DEMO ONLY - Synthetic HGPS images used',
    'dataset': {
        'total_samples': len(df),
        'control_samples': int((df.risk_label == 0).sum()),
        'hgps_samples': int((df.risk_label == 1).sum()),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'hgps_source': 'synthetic (generated from control images)'
    },
    'results': {
        'classical_ml': results,
        'cnn_test_accuracy': test_acc,
        'cnn_test_f1': test_f1,
        'qsvm_test_accuracy': qsvm_acc,
        'qnn_test_accuracy': qnn_acc
    }
}

with open('results/training_report_synthetic.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

print('\n' + '=' * 60)
print('TRAINING COMPLETE')
print('=' * 60)
print(f'Dataset: {len(df)} samples ({(df.risk_label==1).sum()} HGPS, {(df.risk_label==0).sum()} controls)')
print(f'CNN: {test_acc:.1f}% accuracy, {test_f1:.3f} F1')
print(f'QSVM: {qsvm_acc:.1f}% accuracy')
print(f'QNN: {qnn_acc:.1f}% accuracy')
print(f'Best Classical: {max(results.items(), key=lambda x: x[1]["accuracy"])[0]}')
print('\nModels saved to models/ directory')
print('Training report saved to results/training_report_synthetic.json')
print('\n*** REMINDER: These models use synthetic HGPS data ***')
print('*** FOR LOCAL DEMO USE ONLY - Do not upload publicly ***')
