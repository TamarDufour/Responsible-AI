import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import confusion_matrix
import seaborn as sns


# ========== Parameters ==========
print("Setting up paths...")

DATA_DIR = "archive"
META_FILE = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
MODEL_PATH = "resnet18_ham10000.pt"


print("Setting up parameters...")

BATCH_SIZE = 32
LR = 1e-4
NUM_EPOCHS = 10
IMG_SIZE = 224
STEP_SIZE = 5
GAMMA = 0.5


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ========== Dataset Class ==========
print("Setting up dataset...")

class HAM10000Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(sorted(df['dx'].unique()))}
        self.sex_map = {'male': 0, 'female': 1}
        self.images = []
        self.aux_data = []
        self.labels = []

        for _, row in tqdm(self.df.iterrows(), leave=False, desc="Loading images", dynamic_ncols=True):
            image_path = os.path.join(DATA_DIR,
                                      "HAM10000_images_part_1" if os.path.exists(os.path.join(DATA_DIR, "HAM10000_images_part_1", row["image_id"] + ".jpg"))
                                      else "HAM10000_images_part_2",
                                      row["image_id"] + ".jpg")
            image = Image.open(image_path).convert('RGB')
            if transform:
                image = transform(image)
            self.images.append(image)

            age = row['age'] if not pd.isnull(row['age']) else 0
            sex = self.sex_map.get(row['sex'], 0)
            self.aux_data.append(torch.tensor([age / 100.0, sex], dtype=torch.float32))
            self.labels.append(self.label_map[row['dx']])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.aux_data[idx], self.labels[idx]

# ========== Transform ==========
print("Setting up transforms...")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ========== Data Prep ==========
print("Loading and splitting data...")

df = pd.read_csv(META_FILE)
train_val_df, test_df = train_test_split(df, test_size=0.25, stratify=df['dx'], random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df['dx'], random_state=42)


print("Preparing custom DataSets...")

train_dataset = HAM10000Dataset(train_df, transform)
val_dataset = HAM10000Dataset(val_df, transform)
test_dataset = HAM10000Dataset(test_df, transform)


print("Preparing DataLoaders...")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ========== Weighted CE Loss ==========
print("Setting up weighted CE Loss...")

label_counts = train_df['dx'].value_counts().sort_index()
class_weights = 1.0 / torch.tensor(label_counts.values, dtype=torch.float32)
class_weights /= class_weights.sum()  # Normalize
class_weights = class_weights.to(DEVICE)

# ========== Model ==========
print("Setting up model...")

class CustomModel(nn.Module):
    def __init__(self, num_classes, aux_input_dim=2):
        super().__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs + aux_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, aux):
        features = self.base_model(x)
        combined = torch.cat((features, aux), dim=1)
        return self.classifier(combined)


print("Creating model, optimization criterion, optimizer and scheduler...")

model = CustomModel(num_classes=len(label_counts)).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# ========== Training & Eval ==========

def criterion_metric(y_true, y_pred, y_probs):
    y_probs_tensor = torch.tensor(y_probs, dtype=torch.float32, device=DEVICE)
    y_true_tensor = torch.tensor(y_true, dtype=torch.long, device=DEVICE)
    return criterion(y_probs_tensor, y_true_tensor).item()


metrics = {
    "Accuracy": lambda y_true, y_pred, y_probs: accuracy_score(y_true, y_pred),
    "F1 Score": lambda y_true, y_pred, y_probs: f1_score(y_true, y_pred, average='macro'),
    "AUC": lambda y_true, y_pred, y_probs: roc_auc_score(y_true, y_probs, multi_class='ovr') if len(np.unique(y_true)) > 1 else 0,
    "Loss": criterion_metric,
}

print("Setting up training and evaluation functions...")

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    loop = tqdm(loader, leave=False, desc="Training", dynamic_ncols=True)
    for imgs, aux, labels in loop:
        imgs, aux, labels = imgs.to(device), aux.to(device), labels.to(device)
        outputs = model(imgs, aux)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)

def evaluate(model, loader, device, metrics):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    loop = tqdm(loader, leave=False, desc="Evaluating", dynamic_ncols=True)
    with torch.no_grad():
        for imgs, aux, labels in loop:
            imgs, aux = imgs.to(device), aux.to(device)
            outputs = model(imgs, aux)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true += labels.tolist()
            y_pred += preds.cpu().tolist()
            y_probs += probs.cpu().tolist()

    results = {name: fn(y_true, y_pred, y_probs) for name, fn in metrics.items()}
    return results


# ========== Train Loop ==========
print("Starting training loop...")

train_stats, val_stats = [], []
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)

    train_results = evaluate(model, train_loader, DEVICE, metrics)
    val_results = evaluate(model, val_loader, DEVICE, metrics)
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_results['Accuracy']:.4f}, F1: {train_results['F1 Score']:.4f}, AUC: {train_results['AUC']:.4f}")
    print(f"Val   Acc: {val_results['Accuracy']:.4f}, F1: {val_results['F1 Score']:.4f}, AUC: {val_results['AUC']:.4f}")

    scheduler.step()

    train_stats.append(train_results)
    val_stats.append(val_results)

# ========== Visualization ==========
print("Visualizing results...")
epochs = range(1, NUM_EPOCHS + 1)
for metric_name in metrics.keys():
    train_metric_values = [stat[metric_name] for stat in train_stats]
    val_metric_values = [stat[metric_name] for stat in val_stats]

    plt.plot(epochs, train_metric_values, label='Train')
    plt.plot(epochs, val_metric_values, label='Val')
    plt.title(metric_name)
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()



# ========== GradCAM Visualization with Labels ==========
print("Visualizing GradCAM with original images and predictions...")

# Map numeric labels to class names
label_names = [
    "akiec",  # Actinic keratoses
    "bcc",    # Basal cell carcinoma
    "bkl",    # Benign keratosis-like lesions
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic nevi
    "vasc"    # Vascular lesions
]

cam = GradCAM(model=model.base_model, target_layers=[model.base_model.layer4[-1]])
model.eval()

for i in range(5):
    img, aux, label = test_dataset[i]
    input_tensor = img.unsqueeze(0).to(DEVICE)
    aux_tensor = aux.unsqueeze(0).to(DEVICE)

    # Forward pass to get prediction
    with torch.no_grad():
        output = model(input_tensor, aux_tensor)
        pred_label = torch.argmax(output, dim=1).item()

    # Compute CAM
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    cam_image = show_cam_on_image(img.permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True)

    # Plot original + CAM side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img.permute(1, 2, 0).numpy())
    axs[0].set_title(f"True: {label_names[label]}")
    axs[0].axis('off')

    axs[1].imshow(cam_image)
    axs[1].set_title(f"Pred: {label_names[pred_label]}")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


# ========== Save and Load Model ==========

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

model = CustomModel(num_classes=len(label_counts)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully.")

# Get predictions and true labels
y_pred = []
y_true = []

with torch.no_grad():
    for imgs, aux, labels in tqdm(test_loader):
        imgs, aux = imgs.to(DEVICE), aux.to(DEVICE)
        outputs = model(imgs, aux)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Create and plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()