import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


def generate_negative_samples_mixed_multitype(
    x, n_samples, methods=None, mask_prob=0.1, gaussian_alpha=1.0, random_state=None
):
    if random_state is not None:
        np.random.seed(random_state)

    if len(x) < 2 or n_samples <= 0:
        return np.zeros((0, x.shape[1]), dtype="float32")

    minv, maxv = x.min(axis=0), x.max(axis=0)

    if methods is None:
        methods = ["manifold", "inter_cluster", "uniform", "gaussian", "mask"]

    n_modes = len(methods)
    each_type = int(np.ceil(n_samples / n_modes))
    all_anomalies = []

    if "manifold" in methods:
        pairs = x[np.random.choice(len(x), size=(each_type, 2), replace=True)]
        x_a, x_b = pairs[:, 0, :], pairs[:, 1, :]
        alphas = 1.0 + np.random.rand(each_type, 1)
        all_anomalies.append((x_b + alphas * (x_b - x_a)).astype("float32"))

    if "inter_cluster" in methods:
        if len(x) > 50:
            try:
                n_clusters = min(5, len(x) // 20)
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    batch_size=32,
                    random_state=random_state,
                    n_init="auto",
                ).fit(x)
                labels = kmeans.labels_

                indices_a = np.random.choice(len(x), size=each_type, replace=True)
                x_a = x[indices_a]
                labels_a = labels[indices_a]

                cluster_indices = [np.where(labels == k)[0] for k in range(n_clusters)]

                indices_b = []
                for lab in labels_a:
                    available_clusters = [k for k in range(n_clusters) if k != lab]
                    if available_clusters:
                        target_cluster = np.random.choice(available_clusters)
                        target_idx = np.random.choice(cluster_indices[target_cluster])
                        indices_b.append(target_idx)
                    else:
                        indices_b.append(np.random.choice(len(x)))

                x_b = x[np.array(indices_b)]
            except Exception:
                pairs = x[np.random.choice(len(x), size=(each_type, 2), replace=True)]
                x_a, x_b = pairs[:, 0, :], pairs[:, 1, :]
        else:
            pairs = x[np.random.choice(len(x), size=(each_type, 2), replace=True)]
            x_a, x_b = pairs[:, 0, :], pairs[:, 1, :]

        alphas = np.random.rand(each_type, 1)
        all_anomalies.append((x_a * alphas + x_b * (1 - alphas)).astype("float32"))

    if "uniform" in methods:
        all_anomalies.append(
            np.random.uniform(minv, maxv, size=(each_type, x.shape[1])).astype("float32")
        )

    if "gaussian" in methods:
        mu, sigma = x.mean(axis=0), x.std(axis=0)
        sigma = np.where(sigma == 0, 1.0, sigma)
        all_anomalies.append(
            (
                np.random.randn(each_type, x.shape[1]) * (sigma * gaussian_alpha)
                + mu
            ).astype("float32")
        )

    if "mask" in methods:
        base_x = x[np.random.choice(len(x), size=each_type, replace=True)].copy()
        mask = np.random.binomial(1, 1 - mask_prob, size=base_x.shape)
        all_anomalies.append((base_x * mask).astype("float32"))

    if not all_anomalies:
        return np.random.uniform(minv, maxv, size=(n_samples, x.shape[1])).astype("float32")

    anomalies = np.concatenate(all_anomalies, axis=0)
    np.random.shuffle(anomalies)
    return anomalies[:n_samples]


class ViewExpert(nn.Module):
    def __init__(self, max_k=100, input_dim=1, embed_dim=16, hidden_dim=16):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.LeakyReLU(0.2),
        )
        self.pos_embed = nn.Parameter(torch.randn(max_k, embed_dim) * 0.01)
        self.attn_gate = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        _, k, _ = x.shape
        h = self.embed(x)
        eff_k = min(k, self.pos_embed.shape[0])
        h = h[:, :eff_k, :] + self.pos_embed[:eff_k, :].unsqueeze(0)
        attn_scores = self.attn_gate(h)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.sum(h * attn_weights, dim=1)
        score = self.mlp(pooled)
        return score, pooled


class MultiViewMoE(nn.Module):
    def __init__(self, num_views=4, max_k=100, embed_dim=32, hidden_dim=32, dropout=0.0):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                ViewExpert(max_k=max_k, input_dim=1, embed_dim=embed_dim, hidden_dim=hidden_dim)
                for _ in range(num_views)
            ]
        )
        self.gating = nn.Sequential(
            nn.Linear(num_views * embed_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_views),
        )

    def forward(self, x):
        _, v, _, _ = x.shape
        expert_scores_list = []
        expert_embeddings_list = []

        for i in range(v):
            score, emb = self.experts[i](x[:, i, :, :])
            expert_scores_list.append(score)
            expert_embeddings_list.append(emb)

        scores = torch.cat(expert_scores_list, dim=1)
        embeddings = torch.cat(expert_embeddings_list, dim=1)
        weights = self.gating(embeddings)
        temperature = 0.4
        weights = F.softmax(weights / temperature, dim=-1)
        final_score = torch.sum(weights * scores, dim=-1)
        return torch.sigmoid(final_score), weights
