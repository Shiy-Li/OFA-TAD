import os

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, Normalizer, QuantileTransformer, StandardScaler


def split_data(seed, all_train_x, all_train_y, all_test_x, all_test_y, scaler_type):
    np.random.seed(seed)

    train_x = all_train_x
    train_y = all_train_y

    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "std":
        scaler = StandardScaler()
    elif scaler_type == "quan":
        scaler = QuantileTransformer(output_distribution="normal")
    elif scaler_type == "l2":
        scaler = Normalizer(norm="l2")
    elif scaler_type == "none":
        scaler = FunctionTransformer()
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")

    scaler.fit(train_x[train_y == 0])

    train_x = scaler.transform(train_x)

    if all_test_x is None:
        test_x = train_x
        test_y = train_y
    else:
        test_x = scaler.transform(all_test_x)
        test_y = all_test_y

    val_x = train_x
    val_y = train_y

    return (
        train_x.astype("float32"),
        train_y.astype("float32"),
        val_x.astype("float32"),
        val_y.astype("float32"),
        test_x.astype("float32"),
        test_y.astype("float32"),
    )


def load_dataset(dataset, seed, scaler_type, *, data_dir):
    np.random.seed(seed)

    npz_path = os.path.join(data_dir, f"{dataset}.npz")
    mat_path = os.path.join(data_dir, f"{dataset}.mat")

    if os.path.exists(mat_path) and dataset in {"arrhythmia", "glass", "ionosphere", "wbc"}:
        data = loadmat(mat_path)
    elif os.path.exists(npz_path):
        data = np.load(npz_path)
    elif os.path.exists(mat_path):
        data = loadmat(mat_path)
    else:
        raise FileNotFoundError(f"Dataset file not found for {dataset} under {data_dir}")

    label = data["y"].astype("float32").squeeze()
    x = data["X"].astype("float32")

    normal_data = x[label == 0]
    anom_data = x[label == 1]

    num_split = len(normal_data) // 2
    train_x = normal_data[:num_split]
    test_x = np.concatenate([normal_data[num_split:], anom_data], 0)

    train_y = np.zeros(len(train_x))
    test_y = np.concatenate([np.zeros(len(normal_data[num_split:])), np.ones(len(anom_data))])

    return split_data(
        seed,
        all_train_x=train_x,
        all_train_y=train_y,
        all_test_x=test_x,
        all_test_y=test_y,
        scaler_type=scaler_type,
    )
