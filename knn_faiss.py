from collections import defaultdict

import numpy as np

try:
    import faiss
except Exception:
    faiss = None


def _pad_to_k(arr, target_cols, fill_value, dtype=None):
    rows = arr.shape[0]
    cols = arr.shape[1] if arr.size else 0
    if cols >= target_cols:
        return arr[:, :target_cols]
    if dtype is None:
        dtype = arr.dtype if arr.size else type(fill_value)
    pad_shape = (rows, target_cols - cols)
    pad_block = np.full(pad_shape, fill_value, dtype=dtype)
    if cols == 0:
        return pad_block
    return np.hstack((arr, pad_block))


class FaissIndexCache:
    def __init__(self, use_gpu=True, gpu_device=0):
        self.use_gpu = bool(use_gpu)
        self.gpu_device = int(gpu_device)
        self._cache = {}

        self._gpu_enabled = False
        self._gpu_resources = None

        if faiss is not None and self.use_gpu:
            try:
                if (
                    hasattr(faiss, "get_num_gpus")
                    and hasattr(faiss, "StandardGpuResources")
                    and faiss.get_num_gpus() > 0
                ):
                    self._gpu_resources = faiss.StandardGpuResources()
                    self._gpu_enabled = True
            except Exception:
                self._gpu_enabled = False
                self._gpu_resources = None

    @property
    def backend(self):
        if faiss is None:
            return "none"
        return "faiss-gpu" if self._gpu_enabled else "faiss-cpu"

    def get_or_build(self, key, x_index):
        item = self._cache.get(key)
        if item is not None:
            index, n_index = item
            if n_index == x_index.shape[0]:
                return index, True

        if faiss is None:
            raise RuntimeError("faiss is not available in this environment")

        x_index = np.ascontiguousarray(x_index.astype("float32", copy=False))
        d = x_index.shape[1]
        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(x_index)

        if self._gpu_enabled:
            try:
                index = faiss.index_cpu_to_gpu(self._gpu_resources, self.gpu_device, cpu_index)
            except Exception:
                index = cpu_index
        else:
            index = cpu_index

        self._cache[key] = (index, x_index.shape[0])
        return index, False

    def clear(self):
        self._cache.clear()


def find_neighbors_faiss(
    x, neighbor_mask, k, *, cache: FaissIndexCache, cache_key, timers=None, counts=None
):
    x = np.asarray(x)
    mask = np.asarray(neighbor_mask).astype(bool)

    x_index = x[mask]
    n_index = x_index.shape[0]
    if n_index == 0:
        raise ValueError("neighbor_mask contains no True values (no candidates to index).")

    if timers is None:
        timers = defaultdict(float)
    if counts is None:
        counts = defaultdict(int)

    t_fit = None
    index, cache_hit = cache.get_or_build(cache_key, x_index)
    if cache_hit:
        counts["knn_cache_hit"] += 1
    else:
        counts["knn_cache_miss"] += 1
        counts["knn_fit"] += 1

    req_train = min(k + 1, n_index)

    x_index_q = np.ascontiguousarray(x_index.astype("float32", copy=False))
    dist_train_sq, idx_train = index.search(x_index_q, req_train)
    counts["knn_query"] += 1
    dist_train_sq = np.where(idx_train < 0, np.inf, dist_train_sq)
    dist_train = np.sqrt(dist_train_sq)

    if req_train >= 2:
        dist_train = dist_train[:, 1:]
        idx_train = idx_train[:, 1:]
    else:
        dist_train = np.zeros((x_index.shape[0], 0))
        idx_train = np.zeros((x_index.shape[0], 0), dtype=int)

    x_query = x[~mask]
    if x_query.shape[0] > 0:
        req_test = min(k, n_index)
        x_query_q = np.ascontiguousarray(x_query.astype("float32", copy=False))
        dist_test_sq, idx_test = index.search(x_query_q, req_test)
        counts["knn_query"] += 1
        dist_test_sq = np.where(idx_test < 0, np.inf, dist_test_sq)
        dist_test = np.sqrt(dist_test_sq)
    else:
        dist_test = np.zeros((0, 0))
        idx_test = np.zeros((0, 0), dtype=int)

    dist_train = _pad_to_k(dist_train, k, np.inf, dtype=float)
    idx_train = _pad_to_k(idx_train, k, -1, dtype=int)
    dist_test = _pad_to_k(dist_test, k, np.inf, dtype=float)
    idx_test = _pad_to_k(idx_test, k, -1, dtype=int)

    dist = np.vstack((dist_train, dist_test))
    idx = np.vstack((idx_train, idx_test))
    return dist, idx
