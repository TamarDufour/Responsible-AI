# ========== Imports ==========
import os
import random
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, \
    average_precision_score
from sklearn.metrics import confusion_matrix
import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import timm


# ========== Seeds ==========
np.random.seed(55)
torch.manual_seed(55)
torch.cuda.manual_seed_all(55)
torch.backends.cudnn.benchmark = True

# ========== Parameters ==========
print("Setting up paths...")
DATA_DIR = "data"
META_FILE = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

MODEL_PATH = "Trained Melanoma Model.pt"
DATA_SPLIT_FOLDER = "Data splits"
RESULTS_FOLDER = "Results files"



print("Setting up parameters...")

# Do we want to load pre-split data or to split and save new splits?
LOAD_DATA_SPLITS = True

# Constants
TRAIN_VAL_TEST_SPLIT = [0.6, 0.2, 0.2]
IMG_SIZE = 224
DROPOUT = 0.5
CONV_CHANNELS = [2048, 1024, 512, 512]
CONV_KERNELS = [3, 3, 3, 3]
CONV_PADDING = [1, 1, 1, 1]
HIDDEN_LAYERS = [512, 256]
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
STEP_SIZE = 8
GAMMA = 0.5

# Transformations
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.5
ROTATION_DEGREES = 30
RESIZE_SCALE = (0.9, 1.1)
RESIZE_RATIO = (0.8, 1.0)
BRIGHTNESS_JITTER = 0.2
CONTRAST_JITTER = 0.2
SATURATION_JITTER = 0.2
HUE_JITTER = 0.2
RANDOM_ERASING_PROB = 0.2


# Device
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
label_2_description = {
    'mel': "Melanoma",
    'nv': "Melanocytic nevi",
    'NMSC': "Non-melanoma skin cancer"
}


df['label'] = df['dx'].map(label_conversion)
df['description'] = df['label'].map(label_2_description)

label_2_idx = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
idx_2_label = {idx: label for label, idx in label_2_idx.items()}

sex_2_idx = {sex: idx for idx, sex in enumerate(df['sex'].dropna().unique())}
idx_2_sex = {idx: sex for sex, idx in sex_2_idx.items()}

localization_2_idx = {loc: idx for idx, loc in enumerate(df['localization'].dropna().unique())}
idx_2_localization ={idx: localization for localization, idx in localization_2_idx.items()}


# ========== Data Splits ==========
if LOAD_DATA_SPLITS:
    print("Loading data splits...")
    train_df = pd.read_csv(os.path.join(DATA_SPLIT_FOLDER, "train_set.csv"))
    val_df = pd.read_csv(os.path.join(DATA_SPLIT_FOLDER, "val_set.csv"))
    test_df = pd.read_csv(os.path.join(DATA_SPLIT_FOLDER, "test_set.csv"))

else:
    print("Splitting data...")
    train_val_df, test_df = train_test_split(df, test_size=TRAIN_VAL_TEST_SPLIT[2], stratify=df['label'], random_state=83)
    train_df, val_df = train_test_split(train_val_df, test_size=TRAIN_VAL_TEST_SPLIT[1] / (TRAIN_VAL_TEST_SPLIT[0] + TRAIN_VAL_TEST_SPLIT[1]), stratify=train_val_df['label'], random_state=83)

    print("Saving data splits...")
    os.makedirs(DATA_SPLIT_FOLDER, exist_ok=True)
    train_df.to_csv(os.path.join(DATA_SPLIT_FOLDER, "train_set.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_SPLIT_FOLDER, "val_set.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_SPLIT_FOLDER, "test_set.csv"), index=False)

# ========== Dataset Class ==========
print("Setting up dataset...")

class HAM10000Dataset(Dataset):
    def __init__(self, df, base_transform=None, random_transform=None):
        self.base_transform = base_transform
        self.use_random_transform = random_transform is not None
        self.random_transform = random_transform

        self.images, self.labels, self.aux_data = [], [], []
        time.sleep(1)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading images", leave=False):
            img_folder = "HAM10000_images_part_1" if os.path.exists(os.path.join(DATA_DIR, "HAM10000_images_part_1", row['image_id'] + ".jpg")) else "HAM10000_images_part_2"
            image_path = os.path.join(DATA_DIR, img_folder, row['image_id'] + ".jpg")
            image = Image.open(image_path)

            self.images.append(base_transform(image))
            self.labels.append(label_2_idx[row['label']])
            age = row['age'] if not pd.isnull(row['age']) else -1
            sex = sex_2_idx.get(row['sex'], 0)
            loc = localization_2_idx.get(row['localization'], 0)
            self.aux_data.append(tensor([age / 100.0, sex, loc], dtype=torch.float32))
        time.sleep(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.use_random_transform:
            return self.random_transform(self.images[idx]), self.aux_data[idx], self.labels[idx]
        else:
            return self.images[idx], self.aux_data[idx], self.labels[idx]

# ========== Transformation ==========
print("Setting up transforms...")

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

def unnormalize_tensor(tensor_img):
    img_np = tensor_img.cpu().numpy().transpose(1,2,0)
    img_np = (img_np * std[None,None,:]) + mean[None,None,:]
    return np.clip(img_np, 0.0, 1.0).astype(np.float32)


resize_tensor_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean,
        std=std
    )
])

resize_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE))
])

random_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(HORIZONTAL_FLIP_PROB),
    transforms.RandomVerticalFlip(VERTICAL_FLIP_PROB),
    transforms.RandomRotation(degrees=ROTATION_DEGREES),
    transforms.RandomResizedCrop(IMG_SIZE, scale=RESIZE_SCALE, ratio=RESIZE_RATIO),
    transforms.ColorJitter(brightness=BRIGHTNESS_JITTER, contrast=CONTRAST_JITTER, saturation=SATURATION_JITTER, hue=HUE_JITTER),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=mean,
        std=std
    ),
    transforms.RandomErasing(RANDOM_ERASING_PROB)
])

# ========== Datasets & Loaders ==========
print("Preparing Custom-DataSets...")

train_dataset = HAM10000Dataset(train_df, resize_transform, random_transform)
val_dataset = HAM10000Dataset(val_df, resize_tensor_transform)
test_dataset = HAM10000Dataset(test_df, resize_tensor_transform)

print("Preparing DataLoaders...")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ========== Weighted CE Loss ==========
print("Setting up weighted CE Loss...")

label_counts = train_df['label'].value_counts().sort_index()
class_weights = 1.0 / tensor(label_counts.values, dtype=torch.float32)
class_weights /= class_weights.sum()
class_weights = class_weights.to(DEVICE)

# ========== Model ==========
print("Setting up model...")

class Melanoma_Model(nn.Module):
    def __init__(self, num_classes=3, aux_input_dim=3):
        super().__init__()
        base_model = timm.create_model('legacy_xception', pretrained=True, features_only=True)
        self.backbone = base_model
        last_channels = self.backbone.feature_info[-1]['num_chs']
        last_shape = IMG_SIZE // self.backbone.feature_info[-1]['reduction']

        self.conv_blocks = nn.ModuleList()
        conv_channels = [last_channels] + CONV_CHANNELS
        for in_ch, out_ch, kernel, pad in zip(conv_channels[:-1], conv_channels[1:], CONV_KERNELS, CONV_PADDING):
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=pad, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            self.conv_blocks.append(block)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        layers = []
        last_channel = conv_channels[-1]
        prev_dim = last_channel
        for hdim in HIDDEN_LAYERS:
            layers.append(nn.Linear(prev_dim, hdim, bias=False))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=DROPOUT))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(nn.Flatten(), *layers)

    def forward(self, x, aux=None):
        out = self.backbone(x)[-1]

        for block in self.conv_blocks:
            out = block(out)

        out = self.global_pool(out)

        logits = self.classifier(out)
        return logits


print("Creating model, optimization criterion, optimizer and scheduler...")

model = Melanoma_Model(num_classes=len(label_counts)).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# ========== Training & Eval ==========

def criterion_metric(y_true, y_pred, y_probs):
    y_probs = tensor(y_probs, dtype=torch.float32, device=DEVICE)
    y_true = tensor(y_true, dtype=torch.long, device=DEVICE)
    return criterion(y_probs, y_true).item()

def true_positive_rate(y_true, y_pred, y_probs):
    cm = confusion_matrix(y_true, y_pred)
    return np.mean([cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 1 for i in range(len(cm))])

def false_positive_rate(y_true, y_pred, y_probs):
    cm = confusion_matrix(y_true, y_pred)
    return np.mean([
        (cm[:, i].sum() - cm[i, i]) / (cm.sum() - cm[i].sum()) if (cm.sum() - cm[i].sum()) > 0 else 0
        for i in range(len(cm))
    ])

def true_negative_rate(y_true, y_pred, y_probs):
    cm = confusion_matrix(y_true, y_pred)
    # per‐class TNR = TN / (TN + FP)
    tnr = []
    for i in range(cm.shape[0]):
        TP = cm[i,i]
        FN = cm[i,:].sum() - TP
        FP = cm[:,i].sum() - TP
        TN = cm.sum() - (TP+FP+FN)
        tnr.append(TN / (TN+FP) if TN+FP>0 else 0)
    return np.mean(tnr)

def negative_predictive_value(y_true, y_pred, y_probs):
    cm = confusion_matrix(y_true, y_pred)
    # per‐class NPV = TN / (TN + FN)
    npv = []
    for i in range(cm.shape[0]):
        TP = cm[i,i]
        FN = cm[i,:].sum() - TP
        FP = cm[:,i].sum() - TP
        TN = cm.sum() - (TP+FP+FN)
        npv.append(TN / (TN+FN) if TN+FN>0 else 0)
    return np.mean(npv)

metrics = {
    "Accuracy": lambda y_true, y_pred, y_probs: accuracy_score(y_true, y_pred),
    "F1 Score": lambda y_true, y_pred, y_probs: f1_score(y_true, y_pred, average='macro'),
    "AUC": lambda y_true, y_pred, y_probs: roc_auc_score(y_true, y_probs, multi_class='ovr'),
    "Loss": criterion_metric,
    "AP": lambda y_true, y_pred, y_probs: average_precision_score(y_true, np.array(y_probs), average="macro"),
    "Precision": lambda y_true, y_pred, y_probs: precision_score(y_true, y_pred, average='macro', zero_division=0),
    "Recall": lambda y_true, y_pred, y_probs: recall_score(y_true, y_pred, average='macro'),
    "TPR": true_positive_rate,
    "FPR": false_positive_rate,
    "TNR": true_negative_rate,
    "NPV": negative_predictive_value,
}

print("Setting up training and evaluation functions...")

def train_epoch(model, loader, optimizer, criterion, device):
    time.sleep(1)

    model.train()
    total_loss = 0.0
    loop = tqdm(loader, leave=False, desc="Training", dynamic_ncols=True)
    for imgs, aux, labels in loop:
        imgs, aux, labels = imgs.to(device), aux.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs, aux)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    time.sleep(1)

    return total_loss / len(loader)

def evaluate(model, loader, device, metrics):
    time.sleep(1)

    model.eval()
    y_true, y_pred, y_probs = [], [], []
    loop = tqdm(loader, leave=False, desc="Evaluating", dynamic_ncols=True)
    with torch.no_grad():
        for imgs, aux, labels in loop:
            imgs, aux = imgs.to(device, non_blocking=True), aux.to(device, non_blocking=True)
            outputs = model(imgs, aux)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true += labels.tolist()
            y_pred += preds.cpu().tolist()
            y_probs += probs.cpu().tolist()

    time.sleep(1)

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

    train_str = "Train " + " | ".join([f"{key}: {train_results[key]:.4f}" for key in train_results])
    val_str = "Val   " + " | ".join([f"{key}: {val_results[key]:.4f}" for key in val_results])
    print(train_str)
    print(val_str)

    scheduler.step()

    train_stats.append(train_results)
    val_stats.append(val_results)

# ========== Training Process Visualization  ==========
print("Visualizing training and validation metrics...")
os.makedirs(RESULTS_FOLDER, exist_ok=True)
epochs = range(1, NUM_EPOCHS + 1)
for metric_name in metrics.keys():
    train_metric_values = [stat[metric_name] for stat in train_stats]
    val_metric_values = [stat[metric_name] for stat in val_stats]

    plt.plot(epochs, train_metric_values, label='Train')
    plt.plot(epochs, val_metric_values, label='Val')
    plt.title(metric_name)
    plt.xlabel("Epoch")
    plt.xticks(epochs)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_FOLDER, f"{metric_name} over epochs.png"))
    plt.show()


# ========== Evaluation  ==========
print("Evaluating on datasets...")
train_results = evaluate(model, train_loader, DEVICE, metrics)
val_results = evaluate(model, val_loader, DEVICE, metrics)
test_results = evaluate(model, test_loader, DEVICE, metrics)
all_results = {"Train: ": train_results, "Validation: ": val_results, "Test: ": test_results}
for dataset_type, results in all_results.items():
    print(dataset_type + ": | ".join([f"{key}: {results[key]:.4f}" for key in results]))


print("Saving results in CSV file...")
os.makedirs(RESULTS_FOLDER, exist_ok=True)
train_save_df = pd.DataFrame(train_stats)
val_save_df = pd.DataFrame(val_stats)
test_save_df = pd.DataFrame([test_results])

train_save_df.insert(0, "epoch", np.arange(1, NUM_EPOCHS + 1))
val_save_df.insert(0, "epoch", np.arange(1, NUM_EPOCHS + 1))
test_save_df.insert(0, "epoch", "Final")

train_save_df.to_csv(os.path.join(RESULTS_FOLDER, "Train results.csv"), index=False)
val_save_df.to_csv(os.path.join(RESULTS_FOLDER, "Validation results.csv"), index=False)
test_save_df.to_csv(os.path.join(RESULTS_FOLDER, "Test results.csv"), index=False)

# ========== GradCAM Visualization with Labels ==========
print("Visualizing GradCAM with original images and predictions...")

os.makedirs(RESULTS_FOLDER, exist_ok=True)

label_names = [label_2_description[idx_2_label[i]] for i in range(len(idx_2_label))]

cam = GradCAM(model=model, target_layers=[model.conv_blocks[-1][0]])
model.eval()

for cls_idx in range(len(label_names)):
    idxs = [i for i,(_,_,lab) in enumerate(iter(test_dataset)) if lab==cls_idx]
    sample = random.sample(idxs, min(5,len(idxs)))
    for k in sample:
        img, aux, true = test_dataset[k]
        inp, aux_inp = img.unsqueeze(0).to(DEVICE), aux.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(inp, aux_inp)
            probs = torch.softmax(out, dim=1)[0]
            pred = probs.argmax().item()
            conf = probs[pred].item()

        gray = cam(input_tensor=inp)[0]
        orig_img = unnormalize_tensor(img)
        cam_img = show_cam_on_image(orig_img, gray, use_rgb=True)

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
        ax1.imshow(orig_img)
        ax1.set_title(f"True:\n{label_names[true]}", fontsize=10)
        ax1.axis('off')
        ax2.imshow(cam_img)
        ax2.set_title(f"Predicted:\n{label_names[pred]} ({conf:.2f})", fontsize=10)
        ax2.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_FOLDER, f"heatmap label {label_names[true]} -  pred {label_names[pred]} - {k}.png"))
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

os.makedirs(RESULTS_FOLDER, exist_ok=True)
cm = confusion_matrix(y_true, y_pred)
cm_perc = cm.astype(float) / cm.sum(axis=1)[:,None]
labels = [label_2_description[idx_2_label[i]] for i in range(len(cm))]
annot = np.empty_like(cm).astype(object)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i,j] = f"{cm[i,j]}\n{cm_perc[i,j]*100:.2f}%"
sns.heatmap(cm_perc, annot=annot, fmt="", cmap="Blues")
ax = plt.gca()
ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, "confusion matrix test.png"))
plt.show()

cam = GradCAM(model=model, target_layers=[model.conv_blocks[-1][0]])
os.makedirs(RESULTS_FOLDER, exist_ok=True)
n = len(label_names)
fig, axes = plt.subplots(n, n, figsize=(n*3, n*3))
for i in range(n):
    for j in range(n):
        axes[i,j].axis("off")
        idxs = [k for k,(t,p) in enumerate(zip(y_true, y_pred)) if t==i and p==j]
        if not idxs:
            axes[i,j].set_title(f"{label_names[i]}→{label_names[j]}\n(no samples)", fontsize=10)
            continue

        k = idxs[0]
        img, aux, _ = test_dataset[k]
        inp = img.unsqueeze(0).to(DEVICE)
        gray = cam(input_tensor=inp)[0]
        orig_img = unnormalize_tensor(img)
        heatmap = show_cam_on_image(orig_img, gray, use_rgb=True)
        heatmap = heatmap.astype(np.float32) / 255

        combined = np.hstack([orig_img, heatmap])
        axes[i,j].imshow(combined)
        axes[i,j].set_title(f"True: {label_names[i]} →\n→ Predicted: {label_names[j]}", fontsize=10)

plt.savefig(os.path.join(RESULTS_FOLDER, "heatmap confusion matrix test.png"))
plt.tight_layout()
plt.show()

# ========== Save and Load Model ==========

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

model = Melanoma_Model(num_classes=len(label_counts)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
model.eval()
print("Model loaded successfully.")