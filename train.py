import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from preprocessing.dataset_loader import load_dataset
from models.cnn_model import VoiceCloneDetectorCNN

# 1. Load preprocessed features
print("Loading dataset...")
X, y = load_dataset()

if len(X) < 4:
    raise ValueError("Please provide at least 4 audio samples to run a train/test split.")

# Reshape X from (N, 128, 94) to (N, 1, 128, 94) for 2D-CNN channel dimension
X = np.expand_dims(X, axis=1)

# 2. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert arrays to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 3. Model, Loss, Optimizer
device = torch.device("cpu")
model = VoiceCloneDetectorCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
EPOCHS = 5
print("\nStarting training loop...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x).squeeze(1)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch}/{EPOCHS}] - Loss: {avg_loss:.4f}")

# 5. Evaluation Phase
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x).squeeze(1)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).int().tolist()
        
        all_preds.extend(preds)
        all_targets.extend(batch_y.int().tolist())

# Metrics computation
acc = accuracy_score(all_targets, all_preds)
prec = precision_score(all_targets, all_preds, zero_division=0)
rec = recall_score(all_targets, all_preds, zero_division=0)
f1 = f1_score(all_targets, all_preds, zero_division=0)

print("\n--- Evaluation Metrics ---")
print(f"Accuracy : {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall   : {rec * 100:.2f}%")
print(f"F1-Score : {f1 * 100:.2f}%")

# 6. Save Model Checkpoint
torch.save(model.state_dict(), "models/voice_detector_cnn.pth")
print("\nModel saved successfully as models/voice_detector_cnn.pth")