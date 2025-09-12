import os
import pandas as pd
import nibabel as nib
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#init logging file
import logging
logging.basicConfig(filename='headache_ppth_DinoV2.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting ADNI CLVL NL vs AD training script")


#############################################
# 1. Custom ADNI Dataset (Load 3D, Extract 2D)
#############################################


class HeadacheDataset(Dataset):
    def __init__(self, csv_path, split='train', transform=None, num_slices=50):
        """
        Args:
            csv_path: Path to the CSV file containing dataset information
            split: 'train', 'test', or 'val' to filter by split (unused here)
            transform: Optional image transformations
            num_slices: Number of axial slices to extract
        """
        # 1) Load and filter to healthy + MCM only
        df = pd.read_csv(csv_path)
        keep = ['HC', 'HC_MIXED', 'PPTH']
        df = df[df['Class'].isin(keep)].reset_index(drop=True)
        self.data = df

        # 2) Setup mapping
        self.label_map = {
            'HC': 0,
            'HC_MIXED': 0,
            'PPTH': 1
        }

        self.transform = transform
        self.num_slices = num_slices
        self.root_dir = "/data/amciilab/mahfuz/Brain/Region_Excluded_2"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        patient_id = row['ID']
        class_name = row['Class']
        label = self.label_map[class_name]

        # Load volume
        img_path = os.path.join(self.root_dir, f"{patient_id}.nii")
        img_3d = nib.load(img_path).get_fdata()
        img_3d = np.transpose(img_3d, (2, 0, 1))  # (slices, H, W)

        # Center-crop slices
        center = img_3d.shape[0] // 2
        half = self.num_slices // 2
        selected = img_3d[center - half:center + half]

        slices = []
        for sl in selected:
            # Normalize to 0–255
            norm = np.uint8(255 * (sl - sl.min()) / (sl.ptp() + 1e-5))
            img = Image.fromarray(norm).convert('RGB')
            if self.transform:
                img = self.transform(img)
            slices.append(img)

        # (num_slices, C, H, W)
        return torch.stack(slices), label

#########################################
# 2. Prepare Dataset and DataLoaders   #
#########################################

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# full_dataset = ADNI3DTo2DSliceDataset(transform=transform)

# # Split
# indices = list(range(len(full_dataset)))
# labels = [full_dataset.data.loc[i, 'DX'] for i in indices]
# train_idx, temp_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
# temp_labels = [full_dataset.data.loc[i, 'DX'] for i in temp_idx]
# val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=42)

# train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
# val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
# test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
full_dataset = HeadacheDataset(
    csv_path='/scratch/frafsani/DinoV2/fold1.csv',
    transform=transform,
    num_slices=50
)

# — change from 80/10/10 to 80/20 (train/val) —
import random

# 1. Gather all indices by class
all_indices = list(range(len(full_dataset)))
labels = full_dataset.data['Class'].tolist()

ppth_indices     = [i for i in all_indices if labels[i] == 'PPTH']
healthy_indices = [i for i in all_indices if labels[i] in ('HC', 'HC_MIXED')]

# 2. Sample exactly 20 per class for validation
random.seed(42)
val_ppth     = random.sample(ppth_indices, 10)
val_healthy = random.sample(healthy_indices, 10)
val_idx     = val_ppth + val_healthy

# 3. The remaining indices become your training set
train_idx = list(set(all_indices) - set(val_idx))

# 4. (Optional) Shuffle train_idx for randomness
random.shuffle(train_idx)

# 5. Build the subsets and loaders
train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
val_dataset   = torch.utils.data.Subset(full_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)

logging.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
logging.info(
    f"Val class counts: "
    f"PPTH={sum(full_dataset.data.loc[val_idx, 'Class']=='PPTH')}, "
    f"Healthy={sum(full_dataset.data.loc[val_idx, 'Class'].isin(['HC','HC_MIXED']))}"
)

train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
val_dataset   = torch.utils.data.Subset(full_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)

logging.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
logging.info(f"Train labels: {train_dataset.dataset.data.loc[train_idx, 'Class'].value_counts().to_dict()}")
logging.info(f"Val labels:   {val_dataset.dataset.data.loc[val_idx, 'Class'].value_counts().to_dict()}")

#########################################
# 3. Load DINOv2 + Attention + MLP     #
#########################################

backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
backbone.eval()
for param in backbone.parameters():
    param.requires_grad = False

class SliceAttentionAggregator(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, x):  # x: (N, D)
        attn_weights = self.attn(x).squeeze(1)
        attn_weights = torch.softmax(attn_weights, dim=0)
        weighted_avg = torch.matmul(attn_weights, x)
        return weighted_avg

class DINOv2ADNIModel(nn.Module):
    def __init__(self, backbone, embed_dim=128, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.attn = SliceAttentionAggregator(384, 64)
        self.embedding_head = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, slices, return_embedding=False):  # slices: (B, N, C, H, W)
        B, N, C, H, W = slices.shape
        embeddings = []
        for i in range(B):
            features = []
            for img in slices[i]:
                with torch.no_grad():
                    feat = self.backbone(img.unsqueeze(0)).squeeze(0)
                features.append(feat)
            features = torch.stack(features)
            aggregated = self.attn(features)
            embedding = self.embedding_head(aggregated)
            embeddings.append(embedding)
        embeddings = torch.stack(embeddings)
        if return_embedding:
            return embeddings
        return self.classifier(embeddings)


logging.info("DINOv2 + Attention + MLP model setup completed")
#########################################
# 4. Novel Loss Function               #
#########################################

class NovelLoss(nn.Module):
    def __init__(self, temperature=0.07, weight_variance=0.1):
        super(NovelLoss, self).__init__()
        self.temperature = temperature
        self.weight_variance = weight_variance
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, features, labels, logits):
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]

        similarity_matrix = torch.matmul(features, features.t()) / self.temperature
        labels_ = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels_, labels_.t()).float().to(features.device)

        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        sim_logits = similarity_matrix - logits_max.detach()

        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        exp_logits = torch.exp(sim_logits) * logits_mask
        log_prob = sim_logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        contrastive_loss = -mean_log_prob_pos.mean()

        unique_labels = torch.unique(labels_)
        variance_loss = 0.0
        for label in unique_labels:
            class_mask = (labels_ == label).squeeze(1)
            if class_mask.sum() > 1:
                class_features = features[class_mask]
                class_mean = class_features.mean(dim=0, keepdim=True)
                variance_loss += ((class_features - class_mean)**2).mean()
        variance_loss = variance_loss / len(unique_labels)

        ce_loss = self.ce_loss(logits, labels)
        total_loss = ce_loss + contrastive_loss + self.weight_variance * variance_loss
        return total_loss

loss_fn = NovelLoss(temperature=0.07, weight_variance=0.1)

#########################################
# 5. Training Loop                     #
#########################################
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DINOv2ADNIModel(backbone, num_classes=2).to(device)
optimizer = optim.Adam(model.embedding_head.parameters(), lr=1e-3)

num_epochs = 1000
train_losses, val_losses = [], []
best_val_acc = 0.0

for epoch in range(num_epochs):
    # — training (unchanged) —
    model.train()
    running_loss = 0.0
    for slices, labels in train_loader:
        slices, labels = slices.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(slices, return_embedding=True)
        logits     = model.classifier(embeddings)
        loss       = loss_fn(embeddings, labels, logits)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    # — validation with report + AUC logging —
    if (epoch+1) % 10 == 0:
        model.eval()
        val_loss = 0.0
        y_true, y_pred, y_score = [], [], []
        with torch.no_grad():
            for slices, labels in val_loader:
                slices, labels = slices.to(device), labels.to(device)
                embeddings = model(slices, return_embedding=True)
                logits     = model.classifier(embeddings)
                val_loss  += loss_fn(embeddings, labels, logits).item()

                probs = torch.softmax(logits, dim=1)[:, 1]  # probability of class “1”
                _, preds = torch.max(logits, 1)

                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                y_score.extend(probs.cpu().tolist())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(y_true, y_pred)
        val_f1  = f1_score(y_true, y_pred, average='weighted')
        try:
            val_auc = roc_auc_score(y_true, y_score)
        except ValueError:
            val_auc = float('nan')  # if only one class present

        report  = classification_report(y_true, y_pred, target_names=['Healthy','PPTH'])

        logging.info(
            f"Epoch {epoch+1} — "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}, "
            f"Val F1: {val_f1:.4f}, "
            f"Val AUC: {val_auc:.4f}"
        )
        logging.info("Classification Report:\n" + report)
        cm = confusion_matrix(y_true, y_pred)
        logging.info(f"Confusion Matrix:\n{cm}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # — train/val loss plot —
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train/Val Loss')
        plt.legend()
        plt.savefig('Models_new/headache_ppth_dinov2_loss_plot.png')
        plt.close()
        # — checkpoint on best validation accuracy —
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'Models_new/headache_ppth_dinov2_best_val_model.pth')
            logging.info(f"Saved best-val model at epoch {epoch+1} (Acc: {best_val_acc:.4f})")



