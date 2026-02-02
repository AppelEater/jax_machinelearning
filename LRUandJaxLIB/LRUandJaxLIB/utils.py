import numpy as np

import jax
import jax.numpy as jnp

import pickle as pkl

import re
import ast
from pathlib import Path


def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)


def parse_value(val):
    """
    Always return a list:
    - '[0.05]'  → [0.05]
    - '90'      → [90.0]
    """
    parsed = ast.literal_eval(val)
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [float(parsed)]


def parse_filename(filename):
    name = Path(filename).stem  # remove .pkl

    pattern = (
        r"absolute_doppler_waveforms_CNO_(.+?),"
        r"(.+?)_and"
        r"(.+?)_samprate_"
        r"(.+?)_"
        r"(\d+(\.\d+)?)$"
    )

    match = re.match(pattern, name)
    if not match:
        raise ValueError("Filename does not match expected format")

    CNO_list = ast.literal_eval(match.group(1))
    doppler_rate_uncertainty = parse_value(match.group(2))
    doppler_uncertainty = parse_value(match.group(3))
    sampling_rate = float(match.group(4))

    return {
        "sampling_rate": sampling_rate,
        "CNO_list": CNO_list,
        "doppler_uncertainty_list": doppler_uncertainty,
        "doppler_rate_uncertainty": doppler_rate_uncertainty,
    }


def load_data(file_name, batch_size, test_ratio=0.8, encoding="raw", encoding_options=None):
    """
    Load the data from the data file path

    data structure = [(Wave_sequence, target), ... ]

    Parameters
    ----------
    file_name : str
        File name of pickled dataset including full path.
    batch_size : int
        Batch size for returned data.
    test_ratio : float
        Proportion of the dataset to use for training. (0,1]
    encoding : str
        "raw" → return sequences
        "STFT" → apply STFT transform.
    encoding_options : dict
        Only used when encoding="STFT". Expected keys:
            - "window_filter" (str, default "hann")
            - "length" (int, default 256)
            - "hop" (int, default length//2)

    Returns
    -------
    train_sequences, train_labels, test_sequences, test_labels
    """

    # Read data
    with open(file_name, "rb") as f:
        data = pkl.load(f)

    # Manage different formats of input file
    if isinstance(data, list):  # Old format: list of (wave, label)
        # No metadata available, guess from file name and use some defaults
        metadata_from_file = parse_filename(file_name)
        metadata = {
            "sampling_rate": metadata_from_file["sampling_rate"],
            "tone_duration": 3.0,

            "tones": [150, 250, 350, 450, 550, 650, 750, 850],
            "noise": True,
            "num_waveforms_per_class": 6000,

            "CNO_list": metadata_from_file["CNO_list"],
            "doppler_uncertainty_list": metadata_from_file["doppler_uncertainty_list"],
            "doppler_rate_uncertainty": metadata_from_file["doppler_rate_uncertainty"],
        }
        version = 1
        targets = len(data)//1000    # assumes 1000 waveforms per target class (normally used before migration)
    elif isinstance(data, dict):  # New format: dict
        metadata = data.get("metadata", None)
        version = data.get("version", None)
        data = data["waveforms"]
        targets = len(metadata["tones"]) + metadata["noise"]
    else:  # Unknown format
        raise ValueError("Unknown dataset format")

    N_total = len(data)
    N_train = int(test_ratio * N_total)
    N_test = N_total - N_train
    # Resulting shape is [N_total of ( N_samples, 1 )]

    # Shuffle
    perm = np.random.permutation(len(data))
    shuffled_data = [data[i] for i in perm]

    # Split train and test, sequences and labels
    train_sequences = jnp.array([x[0] for x in shuffled_data[:N_train]])
    test_sequences  = jnp.array([x[0] for x in shuffled_data[N_train:]])
    train_labels = one_hot(jnp.array([x[1] for x in shuffled_data[:N_train]]), targets)
    test_labels  = one_hot(jnp.array([x[1] for x in shuffled_data[N_train:]]), targets)
    # Resulting shapes are:
    #  train_sequences : [N_train, N_samples]
    #  test_sequences  : [N_test, N_samples]
    #  train_labels    : [N_train, targets]
    #  test_labels     : [N_test, targets]

    # === ENCODING STEP ===
    if encoding == "Raw":
        train_sequences = train_sequences.reshape((N_train, train_sequences.shape[1], 1))
        test_sequences  = test_sequences.reshape((N_test, test_sequences.shape[1], 1))
        # Resulting shapes are:
        #  train_sequences : [N_train, N_samples, 1]
        #  test_sequences  : [N_test, N_samples, 1]

    elif encoding == "STFT":
        if encoding_options is None:
            encoding_options = {}
        window_filter = encoding_options.get("Window Filter", "hann")
        length = encoding_options.get("Length", 256)
        hop = encoding_options.get("Hop", length // 2)

        train_sequences = jax.vmap(lambda input: jnp.abs(jnp.transpose(jax.scipy.signal.stft(input, fs=2000, window=window_filter, nperseg=length, noverlap = hop, return_onesided=True)[2])))(train_sequences)
        test_sequences = jax.vmap(lambda input: jnp.abs(jnp.transpose(jax.scipy.signal.stft(input, fs=2000, window=window_filter, nperseg=length, noverlap = hop, return_onesided=True)[2])))(test_sequences)
        # Resulting shapes are:
        #  train_sequences : [N_train, roundup(N_samples / Hop) + 1, Hop + 1]
        #  test_sequences  : [N_test, roundup(N_samples / Hop) + 1, Hop + 1]

    else:
        raise ValueError(f"Unknown encoding type: {encoding}")

    # === BATCHING STEP (unified) ===
    train_sequences = train_sequences.reshape((N_train // batch_size, batch_size, *train_sequences.shape[1:]))
    test_sequences  = test_sequences.reshape((N_test  // batch_size, batch_size, *test_sequences.shape[1:]))

    train_labels = train_labels.reshape((N_train // batch_size, batch_size, -1))
    test_labels  = test_labels.reshape((N_test  // batch_size, batch_size, -1))

    return [train_sequences, train_labels, test_sequences, test_labels], metadata
