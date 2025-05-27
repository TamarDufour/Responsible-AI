import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ========== Parameters ==========
print("Setting up paths...")
DATA_DIR = "archive"
META_FILE = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
MODEL_PATH = "resnet18_ham10000.pt"


print("Setting up parameters...")
#set seeds

np.random.seed(55)
torch.manual_seed(55)
torch.cuda.manual_seed_all(55)

# Define constants
TRAIN_VAL_TEST_SPLIT = [0.6, 0.2, 0.2]
IMG_SIZE = 224
DROPOUT = 0.5
HIDDEN_LAYERS = [64, 32]
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
STEP_SIZE = 5
GAMMA = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ========== Load Metadata and Create Label Mappings ==========
print("Loading metadata and setting up label mappings...")
df = pd.read_csv(META_FILE)

label_conversion = {
    'mel': 'mel',
    'nv': 'nv',
    'bcc': 'NMSC',
    'akiec': 'NMSC',
    'bkl': 'NMSC',
    'df': 'NMSC',
    'vasc': 'NMSC'
}
df['dx_grouped'] = df['dx'].map(label_conversion)

label_2_idx = {label: idx for idx, label in enumerate(sorted(df['dx_grouped'].unique()))}
idx_2_label = {idx: label for label, idx in label_2_idx.items()}

sex_2_idx = {sex: idx for idx, sex in enumerate(df['sex'].dropna().unique())}
localization_2_idx = {loc: idx for idx, loc in enumerate(df['localization'].dropna().unique())}

label_2_description = {
    'mel': "Melanoma",
    'nv': "Melanocytic nevi",
    'NMSC': "Non-melanoma skin cancer"
}


# ========== Data Splitting ==========
print("Splitting data...")

train_val_df, test_df = train_test_split(df, test_size=TRAIN_VAL_TEST_SPLIT[2], stratify=df['dx_grouped'], random_state=83)
#Save the test set to a CSV file
test_df.to_csv("test_set.csv", index=False)

train_df, val_df = train_test_split(train_val_df, test_size=TRAIN_VAL_TEST_SPLIT[1] / (TRAIN_VAL_TEST_SPLIT[0] + TRAIN_VAL_TEST_SPLIT[1]), stratify=train_val_df['dx_grouped'], random_state=83)
train_df.to_csv("train_set.csv", index=False)
val_df.to_csv("val_set.csv", index=False)

# ========== Dataset Class ==========
print("Setting up dataset...")

class HAM10000Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.transform = transform
        self.images, self.labels, self.aux_data = [], [], []
        time.sleep(1)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading images", leave=False):
            img_folder = "HAM10000_images_part_1" if os.path.exists(os.path.join(DATA_DIR, "HAM10000_images_part_1", row['image_id'] + ".jpg")) else "HAM10000_images_part_2"
            image_path = os.path.join(DATA_DIR, img_folder, row['image_id'] + ".jpg")
            image = Image.open(image_path)#.convert('RGB')
            self.images.append(transform(image))
            self.labels.append(label_2_idx[label_conversion[row['dx']]])
            age = row['age'] if not pd.isnull(row['age']) else 0
            sex = sex_2_idx.get(row['sex'], 0)
            loc = localization_2_idx.get(row['localization'], 0)
            self.aux_data.append(torch.tensor([age / 100.0, sex, loc], dtype=torch.float32))
        time.sleep(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.aux_data[idx], self.labels[idx]

# ========== Transformation ==========
print("Setting up transforms...")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

print("Preparing custom DataSets...")

train_dataset = HAM10000Dataset(train_df, transform)
val_dataset = HAM10000Dataset(val_df, transform)
test_dataset = HAM10000Dataset(test_df, transform)

test_images = torch.stack([img for img, _, _ in test_dataset])
test_aux = torch.stack([aux for _, aux, _ in test_dataset])
test_labels = torch.tensor([label for _, _, label in test_dataset])

# Save tensors
torch.save({
    'images': test_images,
    'aux': test_aux,
    'labels': test_labels
}, "test_data.pt")

print("Test data saved to test_data.pt")

print("Preparing DataLoaders...")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ========== Weighted CE Loss ==========
print("Setting up weighted CE Loss...")

label_counts = train_df['dx_grouped'].value_counts().sort_index()
class_weights = 1.0 / torch.tensor(label_counts.values, dtype=torch.float32)
class_weights /= class_weights.sum()  # Normalize
class_weights = class_weights.to(DEVICE)

# ========== Model ==========
print("Setting up model...")

class CustomModel(nn.Module):
    def __init__(self, num_classes, aux_input_dim=3):
        super().__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        layers = []
        input_dim = num_ftrs + aux_input_dim
        for hidden_size in HIDDEN_LAYERS:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT))
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x, aux):
        features = self.base_model(x)
        combined = torch.cat((features, aux), dim=1)
        return self.classifier(combined)


print("Creating model, optimization criterion, optimizer and scheduler...")

model = CustomModel(num_classes=len(label_counts)).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# ========== Training & Eval ==========

def criterion_metric(y_true, y_pred, y_probs):
    y_probs = torch.tensor(y_probs, dtype=torch.float32, device=DEVICE)
    y_true = torch.tensor(y_true, dtype=torch.long, device=DEVICE)
    return criterion(y_probs, y_true).item()


metrics = {
    "Accuracy": lambda y_true, y_pred, y_probs: accuracy_score(y_true, y_pred),
    "F1 Score": lambda y_true, y_pred, y_probs: f1_score(y_true, y_pred, average='macro'),
    "AUC": lambda y_true, y_pred, y_probs: roc_auc_score(y_true, y_probs, multi_class='ovr') if len(np.unique(y_true)) > 1 else 0,
    "Loss": criterion_metric,
}

print("Setting up training and evaluation functions...")

def train_epoch(model, loader, optimizer, criterion, device):
    time.sleep(1)

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

    time.sleep(1)

    return sum(losses) / len(losses)

def evaluate(model, loader, device, metrics, results_path):
    time.sleep(1)

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

    time.sleep(1)

    results = {name: fn(y_true, y_pred, y_probs) for name, fn in metrics.items()}
    #save the results to a CSV file
    results_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs
    })
    results_df.to_csv(results_path, index=False)
    return results


# ========== Train Loop ==========
print("Starting training loop...")

train_stats, val_stats = [], []
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)

    train_results = evaluate(model, train_loader, DEVICE, metrics, "Results files/train_results.csv")
    val_results = evaluate(model, val_loader, DEVICE, metrics, "Results files/val_results.csv")

    train_str = "Train " + " | ".join([f"{key}: {train_results[key]:.4f}" for key in train_results])
    val_str = "Val   " + " | ".join([f"{key}: {val_results[key]:.4f}" for key in val_results])
    print(train_str)
    print(val_str)

    scheduler.step()

    train_stats.append(train_results)
    val_stats.append(val_results)

# ========== Training Process Visualization  ==========
print("Visualizing training and validation metrics...")
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


# ========== Evaluation  ==========
print("Evaluating on test set...")
test_results = evaluate(model, test_loader, DEVICE, metrics, "Results files/test_results.csv")
print(f"Test  Acc: {test_results['Accuracy']:.4f}, F1: {test_results['F1 Score']:.4f}, AUC: {test_results['AUC']:.4f}")
#save the results to a CSV file
test_results_df = pd.DataFrame(test_results, index=[0])
test_results_df.to_csv("test_results_metrics.csv", index=False)

# ========== GradCAM Visualization with Labels ==========
print("Visualizing GradCAM with original images and predictions...")

label_names = [label_2_description[idx_2_label[i]] for i in range(len(idx_2_label))]

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
    axs[0].set_title(f"True: {label_names[label]}", fontsize=14)
    axs[0].axis('off')

    axs[1].imshow(cam_image)
    axs[1].set_title(f"Pred: {label_names[pred_label]}", fontsize=14)
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


# ========== Confusion Matrix ==========
print("Generating confusion matrix...")
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, aux, labels in test_loader:
        imgs, aux = imgs.to(DEVICE), aux.to(DEVICE)
        out = model(imgs, aux)
        pred = torch.argmax(out, dim=1)
        y_true.extend(labels.numpy())
        y_pred.extend(pred.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
labels = [label_2_description[idx_2_label[i]] for i in range(len(cm))]
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
# save the confusion matrix as an image
plt.savefig("confusion_matrix_test.png")
plt.show()

# ========== Save and Load Model ==========

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

model = CustomModel(num_classes=len(label_counts)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully.")