import os
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import time
import pickle

version = "1.0"

# -----------------------------
# Signal parameters
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a tone sequence for inference testing"
    )

    # Signal parameters
    parser.add_argument("--sampling-rate", type=float, default=2000,
                        help="Sampling rate [Hz]")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Duration per tone [s]")
    parser.add_argument("--num-tones", type=int, default=50,
                        help="Number of tones in the sequence")
    parser.add_argument("--order", type=int, default=8,
                        help="MFSK order")
    parser.add_argument("--fist-freq", type=int, default=150,
                        help="Frequency of the lowest tome")
    parser.add_argument("--freq-spacing", type=int, default=100,
                        help="Tones frequency spacing")

    # Modulation / channel parameters
    parser.add_argument("--cno", type=float, default=14.2,
                        help="Carrier-to-noise density C/N0 [dB-Hz]")
    parser.add_argument("--doppler-uncertainty", type=float, default=90.0,
                        help="Max Doppler offset [Hz]")
    parser.add_argument("--doppler-rate-uncertainty", type=float, default=16.67,
                        help="Max Doppler rate [Hz/s]")

    # Randomness
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output pickle file")

    return parser.parse_args()

# -------------------------------------------------------------

def main():

    args = parse_args()

    sampling_rate = args.sampling_rate
    duration = args.duration
    num_tones = args.num_tones
    frequencies = jnp.arange(args.order) * args.freq_spacing + args.fist_freq 
    CNO = args.cno
    doppler_uncertainty = args.doppler_uncertainty
    doppler_rate_uncertainty = args.doppler_rate_uncertainty

    # -------------------------------------------------------------

    if args.seed is None:
        key = jax.random.PRNGKey(time.time_ns())
    else:
        key = jax.random.PRNGKey(args.seed)
    key, key_dopp = jax.random.split(key)

    samples_per_tone = int(sampling_rate * duration)

    t = jnp.linspace(0, duration, samples_per_tone, endpoint=False)

    fixed_doppler = jax.random.uniform(key_dopp, shape=(), minval=-doppler_uncertainty / 2, maxval= doppler_uncertainty / 2)

    noise_std = jnp.sqrt(sampling_rate / 2 / 10**(CNO / 10))

    stream = []
    tone_labels = []   # optional, for debugging / evaluation

    for i in range(num_tones):
        key, k_freq, k_phase, k_dru, k_noise = jax.random.split(key, 5)

        # Pick random MFSK symbol
        symbol_idx = jax.random.randint(k_freq, shape=(), minval=0, maxval=8)
        freq = frequencies[symbol_idx]

        # Doppler rate for this tone (varies per tone)
        dru = jax.random.uniform(k_dru, shape=(), minval=-doppler_rate_uncertainty, maxval= doppler_rate_uncertainty)

        # Instantaneous frequency
        freq_t = freq + fixed_doppler + dru * t

        # Random phase
        phase = jax.random.uniform(k_phase, shape=(), minval=0, maxval=2*jnp.pi)

        # Unit-power sinusoid
        wave = jnp.sqrt(2) * jnp.sin(2*jnp.pi*freq_t * t + phase)

        # Add noise
        wave += jax.random.normal(k_noise, shape=(samples_per_tone,)) * noise_std

        # Final normalization (same as training)
        wave /= jnp.sqrt(jnp.mean(wave**2))

        stream.append(wave)
        tone_labels.append(int(symbol_idx))

    stream = np.asarray(jnp.concatenate(stream))
    tone_labels = np.array(tone_labels)

    metadata = {
        "generator": os.path.basename(__file__),
        "generator_version": version,

        "sampling_rate": sampling_rate,
        "tone_duration": duration,

        "tones": np.array(frequencies),
        "noise": False,
        "num_tones": num_tones,

        "CNO_list": CNO,
        "doppler_uncertainty_list": doppler_uncertainty,
        "doppler_rate_uncertainty": doppler_rate_uncertainty,
    }

    output = {
        "version": 1,
        "metadata": metadata,
        "stream": stream,
        "tone_labels": tone_labels
    }

    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"inference_tones_stream_{timestamp}.pkl"

    with open(f"datasets/{args.output}", "wb") as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    main()
