import glob
import pickle as pkl
import numpy as np
import csv
import json

# Optional: skip entire top-level blocks by name
EXCLUDE_KEYS = {"Model Parameters"}  # case-sensitive

try:
    import jax.numpy as jnp
    HAS_JAX = True
except Exception:
    HAS_JAX = False

def to_py(x):
    if isinstance(x, np.generic):
        return x.item()
    if HAS_JAX and isinstance(x, jnp.ndarray):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        return x.item() if x.ndim == 0 else x.tolist()
    return x

def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return to_py(obj)

def flatten(d, parent=""):
    flat = {}
    for k, v in d.items():
        if k in EXCLUDE_KEYS:
            continue
        key = f"{parent}.{k}" if parent else str(k)
        if isinstance(v, dict):
            flat.update(flatten(v, key))
        else:
            flat[key] = v
    return flat

def to_str(v):
    # readable strings for printing/CSV
    if isinstance(v, (list, dict)):
        try:
            return json.dumps(v)
        except Exception:
            return repr(v)
    return str(v)

def build_table(glob_pattern="results/grid_search26/*.pkl"):
    rows = []
    for path in sorted(glob.glob(glob_pattern)):
        with open(path, "rb") as f:
            data = sanitize(pkl.load(f))
        if isinstance(data, dict):
            flat = flatten(data)
        else:
            flat = {"payload": data}
        flat["__path__"] = path
        rows.append(flat)

    # columns: path first, then sorted keys seen across all rows
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    columns = ["__path__"] + sorted(k for k in all_keys if k != "__path__")
    # normalize rows (fill missing with "")
    norm_rows = []
    for r in rows:
        norm_rows.append([to_str(r.get(c, "")) for c in columns])
    return columns, norm_rows

def print_table(columns, rows, max_col_width=60):
    # compute widths
    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], min(len(cell), max_col_width))
    # format helpers
    def trunc(s, w):
        return s if len(s) <= w else s[: max(0, w-1)] + "…"
    def line(parts, widths):
        return " | ".join(trunc(p, w).ljust(w) for p, w in zip(parts, widths))

    sep = "-+-".join("-" * w for w in widths)
    print(line(columns, widths))
    print(sep)
    for r in rows:
        print(line(r, widths))

def save_csv(columns, rows, path="results_table.csv"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"\nSaved CSV to {path}")

# ---------------- main ----------------
if __name__ == "__main__":
    cols, rows = build_table("results/grid_search26/*.pkl")
    print_table(cols, rows)
    save_csv(cols, rows)
