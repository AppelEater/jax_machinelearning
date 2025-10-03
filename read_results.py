# --- ONE CELL: load, name blocks, summarize, and plot ------------------------
import pickle
from pathlib import Path

import numpy as np
try:
    import jax.numpy as jnp
    JAX = True
except Exception:
    jnp = np
    JAX = False

import matplotlib.pyplot as plt
from collections import defaultdict

# -------- 1) Load results.pkl
pkl_path = Path("/root/Project/jax_machinelearning/results/grid_search34/results0 time 2025-10-01 09-27-04.134437.pkl")
with pkl_path.open("rb") as f:
    data = pickle.load(f)

# If your file stores the parameters under "Model Parameters" (as in your screenshot),
# extract them. We assume the list/tuple order is:
# [Encoding_layer, LRU_sub_1, LRU_Mixer_1, LRU_sub_2, LRU_Mixer_2, Decoding_layer]
# Adjust the key or order below if your structure differs.
if isinstance(data, dict) and ("Model Parameters" in data or "model" in data):
    params_seq = data.get("Model Parameters", data.get("model")) # Isolate just the model parameters
else:
    # Some scripts save the raw sequence at the top level
    params_seq = data

# Safety check: we expect 6 top-level blocks
if not isinstance(params_seq, (list, tuple)) or len(params_seq) < 6:
    raise ValueError(
        f"Unexpected structure. Expected a list/tuple with 6 blocks, got: {type(params_seq)} (len={getattr(params_seq,'__len__', lambda: 'NA')()})"
    )

# # -------- 2) Build a named model container for nicer labels
# model = {
#     "Encoding_layer": params_seq[0],
#     "LRU_sub_1":      params_seq[1],
#     "LRU_Mixer_1":    params_seq[2],
#     "LRU_sub_2":      params_seq[3],
#     "LRU_Mixer_2":    params_seq[4],
#     "Decoding_layer": params_seq[5],
# }

# # -------- 3) Utilities to walk any nested param structure
# def _is_array(x):
#     return isinstance(x, (np.ndarray, jnp.ndarray))

# def _size_of(x):
#     return int(np.prod(np.array(x.shape))) if hasattr(x, "shape") else 0

# def iter_params(tree, prefix=""):
#     """
#     Yields (path, array) for every array found in a nested dict/list/tuple.
#     """
#     if _is_array(tree):
#         yield prefix, tree
#     elif isinstance(tree, dict):
#         for k, v in tree.items():
#             child = f"{prefix}.{k}" if prefix else str(k)
#             yield from iter_params(v, child)
#     elif isinstance(tree, (list, tuple)):
#         for i, v in enumerate(tree):
#             child = f"{prefix}[{i}]"
#             yield from iter_params(v, child)
#     else:
#         # scalar or unsupported leaf -> ignore
#         return

# # -------- 4) Collect shapes/sizes and pretty-print
# def summarize_model(model_dict):
#     rows = []
#     per_block_counts = defaultdict(int)
#     per_block_params = defaultdict(int)

#     print("\n=== Parameter Shapes ===")
#     for block_name, block in model_dict.items():
#         found_any = False
#         for path, arr in iter_params(block, prefix=block_name):
#             found_any = True
#             shape = tuple(arr.shape)
#             n = _size_of(arr)
#             per_block_counts[block_name] += 1
#             per_block_params[block_name] += n
#             rows.append((path, shape, n))
#             print(f"{path:<40} shape={shape!s:<20} params={n}")
#         if not found_any:
#             print(f"{block_name:<40} <no arrays found>")

#     total_params = sum(per_block_params.values())
#     total_tensors = sum(per_block_counts.values())
#     print(f"\nTotal tensors: {total_tensors}")
#     print(f"Total parameters: {total_params:,}")

#     return rows, per_block_counts, per_block_params

# rows, per_block_counts, per_block_params = summarize_model(model)

# # -------- 5) Plot: parameters per block (horizontal bar)
# blocks = list(per_block_params.keys())
# values = [per_block_params[b] for b in blocks]

# plt.figure()
# y = np.arange(len(blocks))
# plt.barh(y, values)
# plt.yticks(y, blocks)
# plt.xlabel("Number of parameters")
# plt.title("Parameter count per model block")
# plt.tight_layout()
# plt.show()

# # -------- 6) (Optional) Compact MLP summaries (works if your MLP params use W/b keys)
# def mlp_summary(mlp, name="MLP"):
#     print(f"\n{name} layers:")
#     layer_idx = 0
#     for path, arr in iter_params(mlp, prefix=name):
#         tail = path.split(".")[-1]
#         if tail.lower().endswith("w"):
#             print(f"  Layer {layer_idx}  W: {arr.shape}, params={_size_of(arr)}")
#         elif tail.lower().endswith("b"):
#             print(f"  Layer {layer_idx}  b: {arr.shape}, params={_size_of(arr)}")
#             layer_idx += 1

# # Example (uncomment if applicable):
# # mlp_summary(model["Encoding_layer"], "Encoding_layer")
# # mlp_summary(model["LRU_Mixer_1"],   "LRU_Mixer_1")
# # mlp_summary(model["LRU_Mixer_2"],   "LRU_Mixer_2")
# # mlp_summary(model["Decoding_layer"], "Decoding_layer")
# # -----------------------------------------------------------------------------
