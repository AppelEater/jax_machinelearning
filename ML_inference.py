import pickle as pkl
import jax
import jax.numpy as jnp
import numpy as np


def load_trained_model(model_pkl_path, epoch=-1):
    """
    Load trained model parameters from pickle file.

    epoch = -1 -> last epoch
    """
    with open(model_pkl_path, "rb") as f:
        data = pkl.load(f)

    # Model parameters are stored as a list over epochs
    model_parameters = data["Model Parameters"][epoch]

    # Other useful metadata
    encoding = data["Encoding"]
    encoding_options = data.get("Encoding Options", None)

    return model_parameters, encoding, encoding_options


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

        # [1, T, F]
        return stft_mag[jnp.newaxis, ...]

    else:
        raise ValueError(f"Unknown encoding type: {encoding}")


def run_sliding_inference(
    stream,
    model_parameters,
    encoding,
    encoding_options,
    window_length,
    key
):
    """
    Apply the model to every maximally-overlapping slice of the stream.
    """

    predictions = []
    dropout = 0.0
    training = False

    for i in range(len(stream) - window_length + 1):

        encoded = encode_single_sequence(
            stream[i:i + window_length],
            encoding,
            encoding_options
        )

        logits = model_forward3(
            encoded,
            model_parameters,
            dropout,
            key,
            training
        )

        probs = jax.nn.softmax(logits)

        predictions.append(np.array(probs))

        key, _ = jax.random.split(key)

    return np.squeeze(np.array(predictions), axis=1)


key = jax.random.key(0)

# Load trained model
model_parameters, encoding, encoding_options = load_trained_model(
    "./results/grid_search34/results0.pkl"
)

# Load streaming data
with open("./data/stream.pkl", "rb") as f:
    stream = pkl.load(f)

predictions = run_sliding_inference(
    stream = jnp.asarray(stream),
    model_parameters=model_parameters,
    encoding=encoding,
    encoding_options=encoding_options,
    window_length=stream_window_length,  # MUST match training waveform length
    key=key
)

predicted_classes = np.argmax(predictions, axis=-1)
