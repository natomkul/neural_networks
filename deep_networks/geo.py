CLASSES = [
    "circle",
    "square",
    "triangle",
    "cross",
    "line"
]

import numpy as np
import cv2
import random

IMG_SIZE = 28

def add_noise(img):
    noise = np.random.normal(0, 10, img.shape)
    img = img + noise
    return np.clip(img, 0, 255)

def random_transform(img):
    angle = random.uniform(-15, 15)
    tx = random.randint(-2, 2)
    ty = random.randint(-2, 2)

    M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
    M[:, 2] += [tx, ty]
    return cv2.warpAffine(img, M, (28, 28), borderValue=0)

def draw_shape(label):
    img = np.zeros((28, 28), dtype=np.uint8)
    thickness = random.randint(1, 3)

    if label == 0:  # circle
        cv2.circle(img, (14, 14), random.randint(7, 10), 255, thickness)

    elif label == 1:  # square
        s = random.randint(10, 14)
        cv2.rectangle(img, (14-s//2, 14-s//2),
                            (14+s//2, 14+s//2), 255, thickness)

    elif label == 2:  # triangle
        pts = np.array([
            [14, 4],
            [4, 22],
            [24, 22]
        ])
        cv2.polylines(img, [pts], True, 255, thickness)

    elif label == 3:  # cross
        cv2.line(img, (14, 4), (14, 24), 255, thickness)
        cv2.line(img, (4, 14), (24, 14), 255, thickness)

    elif label == 4:  # line
        if random.random() > 0.5:
            cv2.line(img, (4, 14), (24, 14), 255, thickness)
        else:
            cv2.line(img, (14, 4), (14, 24), 255, thickness)

    img = random_transform(img)
    img = add_noise(img)
    return img

import torch
from torch.utils.data import Dataset

class ShapesDataset(Dataset):
    def __init__(self, n_samples):
        self.n = n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        label = random.randint(0, 4)
        img = draw_shape(label)
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return img, label

import torch.nn as nn

class ShapeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.net(x)

from torch.utils.data import DataLoader
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = ShapesDataset(60000)
test_ds = ShapesDataset(10000)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128)

model = ShapeCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    correct = total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1}: train acc = {acc:.4f}")



