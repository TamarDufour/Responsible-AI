import pandas as pd
import matplotlib.pyplot as plt
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
import ast

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
DATA_DIR = "data" #CHANGE TO YOUR DATA DIRECTORY
META_FILE = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
MODEL_PATH = "resnet18_ham10000.pt"



def load_data(train_df, val_df, test_df):
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
    print("DataLoaders ready.")
    return train_loader, val_loader, test_loader

class HAM10000Dataset(Dataset):
    def __init__(self, df, transform=None):

        self.transform = transform
        self.images, self.labels, self.aux_data = [], [], []
        time.sleep(1)
        sex_2_idx = {sex: idx for idx, sex in enumerate(df['sex'].dropna().unique())}
        localization_2_idx = {loc: idx for idx, loc in enumerate(df['localization'].dropna().unique())}

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

def criterion_metric(y_true, y_pred, y_probs):
    y_probs = torch.tensor(y_probs, dtype=torch.float32, device=DEVICE)
    y_true = torch.tensor(y_true, dtype=torch.long, device=DEVICE)
    return criterion(y_probs, y_true).item()

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
    return results, results_df

def create_confusion_matrix(y_true, y_pred, idx_2_label, title):
    label_2_description = {
        'mel': "Melanoma",
        'nv': "Melanocytic nevi",
        'NMSC': "Non-melanoma skin cancer"
    }

    print("Generating confusion matrix...")

    # Normalize confusion matrix row-wise (true labels)
    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
    labels = [label_2_description[idx_2_label[i]] for i in range(len(cm))]
    annot_labels = np.array([[f"{val:.1f}%" for val in row] for row in cm])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{title} (Percent %)")
    plt.tight_layout()
    plt.savefig(f"{title}_confusion_matrix_percent.png")
    plt.show()

def risk_prediction(y_probs, y_true, risk_threshold=0.15, title = "test"):
        """
        Predict risk based on probabilities.
        If the patient has a melanoma probability greater than the risk threshold, they are considered at risk.
        Calculate the risk, and than create a confusion matrix based on the risk prediction.
        """
        risk_prediction = [1 if prob[0] > risk_threshold else 0 for prob in y_probs]
        y_true_binary = [1 if label == 0 else 0 for label in y_true]

        #create confusion matrix based on risk prediction
        cm_risk = confusion_matrix(y_true_binary, risk_prediction, normalize='true') * 100
        annot_labels = np.array([[f"{val:.1f}%" for val in row] for row in cm_risk])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_risk, annot=annot_labels,fmt='', cmap="BuGn", xticklabels=['Not at risk', 'At risk'], yticklabels=['Not at risk', 'At risk'])
        plt.xlabel("Predicted Risk")
        plt.ylabel("True Risk")
        plt.title(f"{title} Risk Prediction (Percent %)")
        plt.tight_layout()
        plt.savefig(f"{title}_risk_prediction_confusion_matrix_percent.png")
        plt.show()




if __name__ == "__main__":
    print("start")
    #label_2_description = {
    #'mel': "Melanoma",
    #'nv': "Melanocytic nevi",
    #'NMSC': "Non-melanoma skin cancer" }

    print(f"Using device: {DEVICE}")
    MODEL_PATH = "resnet18_ham10000.pt"
    train_df = pd.read_csv("train_set.csv")
    val_df = pd.read_csv("val_set.csv")
    test_df = pd.read_csv("Results files/test_set.csv")
    metadata = pd.read_csv(META_FILE)
    label_conversion = {
        'mel': 'mel',
        'nv': 'nv',
        'bcc': 'NMSC',
        'akiec': 'NMSC',
        'bkl': 'NMSC',
        'df': 'NMSC',
        'vasc': 'NMSC'
    }
    metadata['dx_grouped'] = metadata['dx'].map(label_conversion)
    label_2_idx = {label: idx for idx, label in enumerate(sorted(metadata['dx_grouped'].unique()))}
    idx_2_label = {idx: label for label, idx in label_2_idx.items()}

    train_loader, val_loader, test_loader = load_data(train_df, val_df, test_df)

    print("Setting up weighted CE Loss...")

    label_counts = train_df['dx_grouped'].value_counts().sort_index()
    class_weights = 1.0 / torch.tensor(label_counts.values, dtype=torch.float32)
    class_weights /= class_weights.sum()  # Normalize
    class_weights = class_weights.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    #I want to load the model from pt file
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = CustomModel(num_classes=len(label_2_idx)).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Model file not found. Please train the model first.")
        exit()
    model = model.to(DEVICE)
    print("Model loaded successfully.")
    model.eval()

    metrics = {
        "Accuracy": lambda y_true, y_pred, y_probs: accuracy_score(y_true, y_pred),
        "F1 Score": lambda y_true, y_pred, y_probs: f1_score(y_true, y_pred, average='macro'),
        "AUC": lambda y_true, y_pred, y_probs: roc_auc_score(y_true, y_probs, multi_class='ovr') if len(
            np.unique(y_true)) > 1 else 0,
        "Loss": criterion_metric,
    }

    print("Evaluating on test set...")
    test_results, test_results_df = evaluate(model, test_loader, DEVICE, metrics, "Results files/test_results.csv")


    print(f"Test  Acc: {test_results['Accuracy']:.4f}, F1: {test_results['F1 Score']:.4f}, AUC: {test_results['AUC']:.4f}")
    # save the results to a CSV file
    test_y_true, test_y_pred, test_y_probs = test_results_df['y_true'].tolist(), test_results_df['y_pred'], test_results_df['y_probs'].tolist()
    test_results_df.to_csv("test_results_metrics.csv", index=False)
    print("Creating confusion matrix...")
    create_confusion_matrix(test_y_true, test_y_pred, idx_2_label, "confusion_matrix_test")

    risk_prediction(test_y_probs, test_y_true, risk_threshold=0.4, title = "test")


    print("Evaluating on train set...")
    train_results, train_results_df = evaluate(model, train_loader, DEVICE, metrics, "Results files/train_results.csv")
    print(
        f"Train Acc: {train_results['Accuracy']:.4f}, F1: {train_results['F1 Score']:.4f}, AUC: {train_results['AUC']:.4f}")
    # save the results to a CSV file
    train_y_true, train_y_pred, train_y_probs = train_results_df['y_true'].tolist(), train_results_df['y_pred'], train_results_df['y_probs'].tolist()
    train_results_df.to_csv("train_results_metrics.csv", index=False)
    create_confusion_matrix(train_y_true, train_y_pred, idx_2_label, "confusion_matrix_train")
    risk_prediction(train_y_probs, train_y_true, risk_threshold=0.4, title="train")











