import argparse
import gc
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer

from data import load_dataset
from knn_faiss import FaissIndexCache, find_neighbors_faiss
from metrics import auc_performance, f1_performance
from model import MultiViewMoE, generate_negative_samples_mixed_multitype


warnings.filterwarnings("ignore")


def _backup_and_reset_csv(path: str, columns):
    if path is None:
        return
    if os.path.exists(path):
        ts = time.strftime("%Y%m%d-%H%M%S")
        os.replace(path, f"{path}.{ts}.bak")
    pd.DataFrame(columns=list(columns)).to_csv(path, index=False)


def run_experiment(
    *,
    seed,
    source_datasets,
    target_datasets,
    n_epochs,
    lr,
    wd,
    scaler_types,
    data_dir,
    embed_dim=32,
    hidden_dim=16,
    max_k_limit=80,
    detailed_output_file=None,
    save_detailed=True,
    batch_size=8192,
    use_gpu_knn=True,
    gpu_device=0,
    verbose=True,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)

    net = MultiViewMoE(
        num_views=len(scaler_types), max_k=max_k_limit, embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=0.0
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    criterion_pred = nn.MSELoss()

    knn_cache = FaissIndexCache(use_gpu=use_gpu_knn, gpu_device=gpu_device)

    if verbose:
        print(f"\n[{time.strftime('%H:%M:%S')}] === Running Seed {seed} ===")
        print(f"Device: {device}")
        print(f"KNN backend: {knn_cache.backend}")

    dataset_cache = {}
    neg_cache = {}
    feat_cache = {}

    def _get_dataset(dataset_name, scaler_type):
        k = (dataset_name, seed, scaler_type)
        if k in dataset_cache:
            return dataset_cache[k]
        out = load_dataset(dataset_name, seed, scaler_type, data_dir=data_dir)
        dataset_cache[k] = out
        return out

    def _get_neg_samples(dataset_name, scaler_type, x_train, n_neg, i):
        k = (dataset_name, seed, scaler_type, n_neg)
        if k in neg_cache:
            return neg_cache[k]
        neg = generate_negative_samples_mixed_multitype(
            x_train,
            n_neg,
            methods=["mask", "manifold", "inter_cluster", "uniform", "gaussian"],
            random_state=seed + i,
        )
        neg_cache[k] = neg
        return neg

    def _get_train_feat(dataset_name, scaler_type, train_x, n_neg, eff_k, neighbor_mask, i, n_train):
        k = (dataset_name, seed, scaler_type, n_train, n_neg, eff_k)
        if k in feat_cache:
            return feat_cache[k]

        neg_x = _get_neg_samples(dataset_name, scaler_type, train_x, n_neg, i)
        x_all = np.vstack([train_x, neg_x])

        dist, _ = find_neighbors_faiss(
            x_all, neighbor_mask, eff_k, cache=knn_cache, cache_key=("train", dataset_name, scaler_type, seed)
        )

        scaler_abs = QuantileTransformer(output_distribution="uniform")
        valid_train_dist = dist[:n_train].flatten()
        valid_train_dist = valid_train_dist[np.isfinite(valid_train_dist)]
        if len(valid_train_dist) > 0:
            scaler_abs.fit(valid_train_dist.reshape(-1, 1))

        abs_dist_norm = np.full_like(dist, 1.0)
        mask_finite = np.isfinite(dist)
        if np.any(mask_finite):
            abs_dist_norm[mask_finite] = scaler_abs.transform(dist[mask_finite].reshape(-1, 1)).flatten()

        feat = np.stack([abs_dist_norm], axis=-1).astype("float32", copy=False)
        feat_cache[k] = feat
        return feat

    train_batch_cache = {}
    if verbose:
        print(f"--- Precomputing Training Features (Seed {seed}) ---")
    for dataset_name in source_datasets:
        data_versions = {st: _get_dataset(dataset_name, st) for st in scaler_types}
        train_x_ref_full = data_versions[scaler_types[0]][0]
        n_train_full = int(train_x_ref_full.shape[0])

        if n_train_full < 2:
            if verbose:
                print(f"[WARN] Skip source dataset {dataset_name}: not enough normal training samples (n={n_train_full}).")
            continue

        idx = np.arange(n_train_full, dtype=int)
        n_train = int(n_train_full)
        eff_k = min(max_k_limit, n_train - 1)
        n_neg = n_train

        y_all = np.hstack([np.zeros(n_train), np.ones(n_neg)]).astype("float32", copy=False)
        neighbor_mask = np.hstack([np.ones(n_train), np.zeros(n_neg)]).astype(bool)

        view_feats = []
        for i, st in enumerate(scaler_types):
            train_x_view = data_versions[st][0][idx]
            feat = _get_train_feat(dataset_name, st, train_x_view, n_neg, eff_k, neighbor_mask, i, n_train)
            view_feats.append(feat)

        x_feat = np.stack(view_feats, axis=1).astype("float32", copy=False)
        train_batch_cache[dataset_name] = (x_feat, y_all)

    if verbose:
        print(f"--- Training (Seed {seed}) ---")

    for epoch in range(n_epochs):
        net.train()
        shuffled_sources = np.random.permutation(source_datasets)
        for dataset_name in shuffled_sources:
            x_feat, y_all = train_batch_cache[dataset_name]
            x_tensor = torch.tensor(x_feat, dtype=torch.float32, device=device)
            y_tensor = torch.tensor(y_all, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            scores, _ = net(x_tensor)
            loss_pred = criterion_pred(scores, y_tensor)
            total_loss = loss_pred
            total_loss.backward()
            optimizer.step()

        if verbose and ((epoch + 1) == 1 or (epoch + 1) == n_epochs or (epoch + 1) % 5 == 0):
            print(f"  Epoch {epoch + 1}/{n_epochs} {loss_pred.item()} done")

    net.eval()

    headers = ["Dataset", "Seed", "AUROC", "AUPRC", "F1"]
    if detailed_output_file is not None and save_detailed:
        parent_dir = os.path.dirname(str(detailed_output_file))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        if not os.path.exists(detailed_output_file):
            pd.DataFrame(columns=headers).to_csv(detailed_output_file, index=False)

    auroc_list = []
    aupr_list = []
    f1_list = []

    if verbose:
        print(f"--- Evaluating (Seed {seed}) ---")

    for dataset_name in target_datasets:
        try:
            data_versions = {
                st: load_dataset(dataset_name, seed, scaler_type=st, data_dir=data_dir)
                for st in scaler_types
            }

            train_x_ref, _, _, _, test_x_ref, test_y_ref = data_versions[scaler_types[0]]
            n_train = int(train_x_ref.shape[0])
            n_test = int(test_x_ref.shape[0])
            eff_k = min(max_k_limit, n_train - 1)

            view_inputs_test = []
            for st in scaler_types:
                train_x, _, _, _, test_x, _ = data_versions[st]
                x_query = np.vstack([train_x, test_x])
                mask_candidates = np.hstack([np.ones(len(train_x)), np.zeros(len(test_x))]).astype(bool)

                dist_all, _ = find_neighbors_faiss(
                    x_query,
                    mask_candidates,
                    eff_k,
                    cache=knn_cache,
                    cache_key=("eval", dataset_name, st, seed),
                )

                dist_train = dist_all[:n_train]
                scaler_abs = QuantileTransformer(output_distribution="uniform")
                valid_d = dist_train.flatten()
                valid_d = valid_d[np.isfinite(valid_d)]
                if len(valid_d) > 0:
                    scaler_abs.fit(valid_d.reshape(-1, 1))

                abs_norm = np.full_like(dist_all, 1.0)
                mask_f = np.isfinite(dist_all)
                if np.any(mask_f):
                    abs_norm[mask_f] = scaler_abs.transform(dist_all[mask_f].reshape(-1, 1)).flatten()

                abs_norm_test = abs_norm[n_train:]
                feat_test = np.stack([abs_norm_test], axis=-1)
                view_inputs_test.append(feat_test)

            x_test_np = np.stack(view_inputs_test, axis=1).astype("float32", copy=False)

            test_scores_parts = []
            with torch.no_grad():
                for start in range(0, n_test, batch_size):
                    end = min(n_test, start + batch_size)
                    x_batch = torch.tensor(x_test_np[start:end], dtype=torch.float32, device=device)
                    scores_batch, _ = net(x_batch)
                    test_scores_parts.append(scores_batch.detach().cpu().numpy())
                    del x_batch, scores_batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            test_scores = (
                np.concatenate(test_scores_parts, axis=0) if test_scores_parts else np.zeros((0,))
            )

            if len(np.unique(test_y_ref)) > 1:
                auroc, auprc = auc_performance(test_scores, test_y_ref)
                best_f1 = f1_performance(test_scores, test_y_ref)
            else:
                auroc, auprc, best_f1 = 0.5, 0.0, 0.0

            if verbose:
                print(
                    f"  > {dataset_name}: AUROC={float(auroc):.4f}, AUPRC={float(auprc):.4f}, F1={float(best_f1):.4f}"
                )

            row = {
                "Dataset": dataset_name,
                "Seed": seed,
                "AUROC": float(auroc),
                "AUPRC": float(auprc),
                "F1": float(best_f1),
            }

            if detailed_output_file is not None and save_detailed:
                pd.DataFrame([row]).to_csv(detailed_output_file, mode="a", header=False, index=False)

            auroc_list.append(float(auroc))
            aupr_list.append(float(auprc))
            f1_list.append(float(best_f1))

            knn_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            if verbose:
                print(f"Error eval {dataset_name} Seed {seed}: {e}")
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            knn_cache.clear()

    avg_auroc = float(np.mean(auroc_list)) if auroc_list else float("nan")
    avg_auprc = float(np.mean(aupr_list)) if aupr_list else float("nan")
    avg_f1 = float(np.mean(f1_list)) if f1_list else float("nan")

    if verbose:
        print(f" > AVG AUROC={avg_auroc:.6f}, AVG AUPRC={avg_auprc:.6f}, AVG F1={avg_f1:.6f}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "seed": int(seed),
        "avg_auroc": avg_auroc,
        "avg_auprc": avg_auprc,
        "avg_f1": avg_f1,
        "n_targets": int(len(auroc_list)),
    }


def generate_final_summary(detailed_file, summary_file, *, seeds=None):
    if not os.path.exists(detailed_file):
        print(f"No detailed results found at {detailed_file}")
        return

    df = pd.read_csv(detailed_file)
    if df.empty:
        print("Detailed results file is empty.")
        return

    if seeds is not None:
        seeds_set = set(int(s) for s in seeds)
        df = df[df["Seed"].astype(int).isin(seeds_set)]
        if df.empty:
            print("Detailed results file has no rows for the provided seeds; summary not generated.")
            return

    summary = df.groupby("Dataset").agg(
        {"AUROC": ["mean", "std"], "AUPRC": ["mean", "std"], "F1": ["mean", "std"]}
    )
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    summary = summary.rename(
        columns={
            "AUROC_mean": "AUROC_Mean",
            "AUROC_std": "AUROC_Std",
            "AUPRC_mean": "AUPRC_Mean",
            "AUPRC_std": "AUPRC_Std",
            "F1_mean": "F1_Mean",
            "F1_std": "F1_Std",
        }
    )

    avg_row = pd.DataFrame(
        {
            "Dataset": ["AVERAGE"],
            "AUROC_Mean": [summary["AUROC_Mean"].mean()],
            "AUROC_Std": [summary["AUROC_Std"].mean()],
            "AUPRC_Mean": [summary["AUPRC_Mean"].mean()],
            "AUPRC_Std": [summary["AUPRC_Std"].mean()],
            "F1_Mean": [summary["F1_Mean"].mean()],
            "F1_Std": [summary["F1_Std"].mean()],
        }
    )
    summary = pd.concat([summary, avg_row], ignore_index=True)

    if summary_file is None:
        print(summary.tail(1))
        return

    if os.path.exists(summary_file):
        ts = time.strftime("%Y%m%d-%H%M%S")
        os.replace(summary_file, f"{summary_file}.{ts}.bak")

    summary.to_csv(summary_file, index=False)
    print(f"\nSummary saved to {summary_file}")
    print(summary.tail(1))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default='./Data/')
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--n_epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=2e-5)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--max_k", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--use_gpu_knn", action="store_true")
    p.add_argument("--gpu_device", type=int, default=0)
    p.add_argument("--output_csv", type=str, default="results_detailed.csv")
    p.add_argument("--summary_csv", type=str, default="results_summary.csv")
    p.add_argument("--quiet", action="store_true")

    args = p.parse_args()

    all_datasets = [
        "amazon", "lympho", "wine", "speech", "yeast", "WPBC", "vertebral", "fault", "Cardiotocography",
        "Wilt", "Hepatitis", "glass", "campaign", "Parkinson", "ionosphere", "mammography", "abalone",
        "satellite", "vowels", "mnist", "cardio", "annthyroid", "imgseg", "thyroid", "wbc", "optdigits",
        "satimage-2", "WDBC", "pendigits", "shuttle", "musk", "breastw", "comm.and.crime", "SpamBase",
        "pima", "arrhythmia", "donors", "backdoor", "cover", "fraud", "census",
    ]

    source_datasets = ["vertebral", "annthyroid", "comm.and.crime", "Cardiotocography", "wine", "imgseg", "satellite"]
    target_datasets = [d for d in all_datasets if d not in source_datasets]

    scaler_types = ["none", "std", "minmax", "quan"]

    headers = ["Dataset", "Seed", "AUROC", "AUPRC", "F1"]
    _backup_and_reset_csv(args.output_csv, headers)

    verbose = not bool(args.quiet)
    if verbose:
        print("--- Starting OFA-TAD Multi-Seed Experiment ---")
        print(f"Seeds: {args.seeds}")
        print(f"Saving detailed results to: {args.output_csv}")
        print(f"Saving summary results to: {args.summary_csv}")

    for seed in args.seeds:
        run_experiment(
            seed=seed,
            source_datasets=source_datasets,
            target_datasets=target_datasets,
            n_epochs=args.n_epochs,
            lr=args.lr,
            wd=args.wd,
            scaler_types=scaler_types,
            data_dir=args.data_dir,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            max_k_limit=args.max_k,
            detailed_output_file=args.output_csv,
            save_detailed=True,
            batch_size=args.batch_size,
            use_gpu_knn=args.use_gpu_knn,
            gpu_device=args.gpu_device,
            verbose=verbose,
        )

    generate_final_summary(args.output_csv, args.summary_csv, seeds=args.seeds)


if __name__ == "__main__":
    main()
