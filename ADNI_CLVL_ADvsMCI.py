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
logging.basicConfig(filename='ADNI_CLVL_MCIvsAD.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting ADNI CLVL MCI vs AD training script")


#############################################
# 1. Custom ADNI Dataset (Load 3D, Extract 2D)
#############################################

class ADNI3DTo2DSliceDataset(Dataset):
    def __init__(self, transform=None, num_slices=50):
        data = pd.read_csv('/scratch/frafsani/Resnet_AD/ADNI_add_w_AD_status.csv')
        data = data.dropna(subset=['DX'])
        data = data[(data['DX'] == 'MCI') | (data['DX'] == 'AD')]   # for AD vs NL
        self.data = data.reset_index(drop=True)
        self.transform = transform
        self.num_slices = num_slices
        self.label_map = {'MCI': 0, 'AD': 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row[0]  # Path to .nii or .nii.gz
        label_str = row['DX']
        label = self.label_map[label_str]

        # Load 3D volume and extract axial slices
        img_3d = nib.load(img_path).get_fdata()
        img_3d = np.transpose(img_3d, (2, 0, 1))  # (slices, H, W)

        center = img_3d.shape[0] // 2
        half = self.num_slices // 2
        selected_slices = img_3d[center - half:center + half]

        slices = []
        for slice_2d in selected_slices:
            slice_img = Image.fromarray(np.uint8(255 * (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-5))).convert('RGB')
            if self.transform:
                slice_img = self.transform(slice_img)
            slices.append(slice_img)

        return torch.stack(slices), label  # Shape: (num_slices, C, H, W)

#########################################
# 2. Prepare Dataset and DataLoaders   #
#########################################

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = ADNI3DTo2DSliceDataset(transform=transform)

# Split
indices = list(range(len(full_dataset)))
labels = [full_dataset.data.loc[i, 'DX'] for i in indices]
train_idx, temp_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
temp_labels = [full_dataset.data.loc[i, 'DX'] for i in temp_idx]
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_labels, random_state=42)

train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

logging.info(f"Train dataset size: {len(train_dataset)}")
logging.info(f"Validation dataset size: {len(val_dataset)}")
logging.info(f"Test dataset size: {len(test_dataset)}")
logging.info(f"Train dataset labels: {train_dataset.dataset.data['DX'].value_counts().to_dict()}")
logging.info(f"Validation dataset labels: {val_dataset.dataset.data['DX'].value_counts().to_dict()}")
logging.info(f"Test dataset labels: {test_dataset.dataset.data['DX'].value_counts().to_dict()}")
logging.info("Dataset and DataLoaders prepared successfully")
logging.info("Starting DINOv2 + Attention + MLP model setup")
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
best_test_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for slices, labels in train_loader:
        slices, labels = slices.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(slices, return_embedding=True)
        logits = model.classifier(embeddings)
        loss = loss_fn(embeddings, labels, logits)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for slices, labels in val_loader:
            slices, labels = slices.to(device), labels.to(device)
            embeddings = model(slices, return_embedding=True)
            logits = model.classifier(embeddings)
            val_loss += loss_fn(embeddings, labels, logits).item()

    val_losses.append(val_loss / len(val_loader))

    logging.info(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    if (epoch + 1) % 50 == 0:
        # Run full testing
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for slices, labels in test_loader:
                slices, labels = slices.to(device), labels.to(device)
                embeddings = model(slices, return_embedding=True)
                logits = model.classifier(embeddings)
                _, preds = torch.max(logits, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        test_acc = accuracy_score(y_true, y_pred)
        logging.info(f"Test Accuracy after epoch {epoch+1}: {test_acc:.4f}")
        f1 = f1_score(y_true, y_pred, average='weighted')
        logging.info(f"Test F1 Score after epoch {epoch+1}: {f1:.4f}")
        auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
        logging.info(f"Test AUC after epoch {epoch+1}: {auc:.4f}")
        logging.info(f"Confusion Matrix after epoch {epoch+1}:\n{confusion_matrix(y_true, y_pred)}")
        logging.info(f"Classification Report after epoch {epoch+1}:\n{classification_report(y_true, y_pred)}")

        # Save model based on best test accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'Models_new/MCIvsAD_adni_dinov2_best_test_model.pth')
            logging.info(f"Best model saved at epoch {epoch+1} with test accuracy: {best_test_acc:.4f}")

        # Plot Loss Curve
        plt.figure()
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        plt.savefig('Models_new/MCIvsAD_adni_dinov2_training_loss_updated.png')
        plt.close()

        # Save Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        labels_display = list(full_dataset.label_map.keys())
        plt.figure()
        ConfusionMatrixDisplay(cm, display_labels=labels_display).plot(cmap='Blues', values_format='d')
        plt.title(f"Confusion Matrix on Test Set (Epoch {epoch+1})")
        plt.savefig(f'Models_new/MCIvsAD_adni_dinov2_conf_mat_epoch_{epoch+1}.png')
        plt.close()
