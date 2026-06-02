from LRUandJaxLIB import *

import argparse
import pickle as pkl
from pathlib import Path
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np

# -------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on an input dataset or signal using given trained model"
    )

    # Input file names
    parser.add_argument("data_file")
    parser.add_argument("model_file")

    return parser.parse_args()

# -------------------------------------------------------------

def encode_single_sequence(raw_sequence, encoding, encoding_options):
    """
    Encode ONE waveform slice exactly like load_data(), without batching.
    """

    if encoding == "Raw":
        # [1, N, 1]
        return raw_sequence.reshape((1, raw_sequence.shape[0], 1))

    elif encoding == "STFT":
        if encoding_options is None:
            encoding_options = {}

        window_filter = encoding_options.get("Window Filter", "hann")
        length = encoding_options.get("Length", 256)
        hop = encoding_options.get("Hop", length // 2)

        _, _, Zxx = jax.scipy.signal.stft(
            raw_sequence,
            fs=2000,
            window=window_filter,
            nperseg=length,
            noverlap=hop,
            return_onesided=True
        )

        stft_mag = jnp.abs(jnp.transpose(Zxx))

        return stft_mag

    else:
        raise ValueError(f"Unknown encoding type: {encoding}")

# -------------------------------------------------------------

def make_encode_and_forward(encoding, encoding_options, model_parameters):
    key = jax.random.PRNGKey(0)

    @jax.jit
    def _encode_and_forward(raw_window):
        encoded = encode_single_sequence(raw_window, encoding, encoding_options)
        logits = model_forward(encoded, model_parameters, prob=0.0, key=key)
        return jax.nn.softmax(logits)

    return _encode_and_forward


def run_sliding_inference(
    stream,
    model_parameters,
    encoding,
    encoding_options,
    window_length,
):
    """
    Apply the model to every maximally-overlapping slice of the stream.
    """

    predictions = []
    encode_and_forward = make_encode_and_forward(encoding, encoding_options, model_parameters)

    for i in tqdm(range(len(stream) - window_length + 1), desc="Iterations"):

        raw_window = jnp.array(stream[i:i + window_length])
        probs = encode_and_forward(raw_window)

        predictions.append(np.array(probs))

    return np.array(predictions)

# -------------------------------------------------------------

def main():

    args = parse_args()

    # Load trained model
    with open(args.model_file, "rb") as f:
        data = pkl.load(f)
    model_parameters = data["parameters"][-1]
    config = data["config"]

    # Load input data
    with open(args.data_file, "rb") as f:
        data = pkl.load(f)

    metadata = data["metadata"]
    stream = data["stream"]
    expected_classes = data["tone_labels"]

    # ---------------------------------------------------------

    hard_keys = ("sampling_rate", "tone_duration", "tones")
    soft_keys = ("CNO_list", "doppler_uncertainty_list", "doppler_rate_uncertainty")    # Lists of values

    for item in hard_keys:
        if metadata[item] != config["dataset"][item]:
            raise ValueError("Input data not compatible with trained model")

    for item in soft_keys:
        if metadata[item] not in config["dataset"][item]:
            print(f"Different values in {item} among input data and trained model")

    stream_window_length = round(metadata["sampling_rate"] * metadata["tone_duration"])

    # ---------------------------------------------------------

    predictions = run_sliding_inference(
        stream = jnp.asarray(stream),
        model_parameters=model_parameters,
        encoding=config["encoding"],
        encoding_options=config["encoding_options"],
        window_length=stream_window_length,
    )

    out_path = Path(args.data_file).stem + '_result.pkl'
    with open(out_path, 'wb') as f:
        pkl.dump(predictions, f)
    print(f"Saved predictions to {out_path}")

# -------------------------------------------------------------

if __name__ == "__main__":
    main()
