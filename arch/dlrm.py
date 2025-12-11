import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchrec.models.dlrm import DLRM
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from sklearn.metrics import roc_auc_score, accuracy_score


class DLRMRecommender:
    def __init__(
        self,
        num_users: int,
        num_movies: int,
        num_genres: int,
        embedding_dim: int = 64,
        dense_arch_layer_sizes: list[int] = [128, 64],
        over_arch_layer_sizes: list[int] = [128, 64, 1],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device

        eb_configs = [
            EmbeddingBagConfig(
                name="userid",
                embedding_dim=embedding_dim,
                num_embeddings=num_users,
                feature_names=["userid"],
            ),
            EmbeddingBagConfig(
                name="movieid",
                embedding_dim=embedding_dim,
                num_embeddings=num_movies,
                feature_names=["movieid"],
            ),
            EmbeddingBagConfig(
                name="genres",
                embedding_dim=embedding_dim,
                num_embeddings=num_genres,
                feature_names=["genres"],
            ),
        ]

        embedding_bag_collection = EmbeddingBagCollection(
            tables=eb_configs,
            device=torch.device(device),
        )

        self.model = DLRM(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=1,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
            dense_device=device,
        ).to(device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.001, weight_decay=1e-5
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        num_batches = 0

        for sparse_features, dense_features, labels in tqdm(
            train_loader, total=len(train_loader)
        ):
            sparse_features = sparse_features.to(self.device)
            dense_features = dense_features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(dense_features, sparse_features)
            loss = self.criterion(logits.squeeze(), labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for sparse_features, dense_features, labels in val_loader:
                sparse_features = sparse_features.to(self.device)
                dense_features = dense_features.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(dense_features, sparse_features)
                loss = self.criterion(logits.squeeze(), labels)

                preds = torch.sigmoid(logits.squeeze())

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()
                num_batches += 1

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        auc = roc_auc_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))

        return {"loss": total_loss / num_batches, "auc": auc, "accuracy": acc}
