import os
import argparse

import numpy as np
import time
import pickle as pkl

version = "1.0"

# -----------------------------
# Signal parameters
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a tone sequence for inference testing"
    )

    # Output
    parser.add_argument("output_file", type=str,
                        help="Output pickle file")

    # Reference trained model
    parser.add_argument("--ref-model", type=str, default=None,
                        help="Sampling rate [Hz]")

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

    return parser.parse_args()

# -------------------------------------------------------------

def main():

    args = parse_args()

    if args.ref_model is None:
        # If no reference model provided, uses default or provided parameters
        sampling_rate = args.sampling_rate
        duration = args.duration
        frequencies = [args.fist_freq + i * args.freq_spacing for i in range(args.order)]
        cno = args.cno
        doppler_uncertainty = args.doppler_uncertainty
        doppler_rate_uncertainty = args.doppler_rate_uncertainty
    else:
        # Otherwise, load the model and retrieve parameters from it
        with open(args.ref_model, "rb") as f:
            data = pkl.load(f)
            config = data["config"]
            sampling_rate = config["dataset"]["sampling_rate"]
            duration = config["dataset"]["tone_duration"]
            frequencies = config["dataset"]["tones"]
            cno = min(config["dataset"]["CNO_list"])
            doppler_uncertainty = max(config["dataset"]["doppler_uncertainty_list"])
            doppler_rate_uncertainty = max(config["dataset"]["doppler_rate_uncertainty"])

    num_tones = args.num_tones

    # -------------------------------------------------------------

    if args.seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(args.seed)

    samples_per_tone = int(sampling_rate * duration)

    t = np.linspace(0, duration, samples_per_tone, endpoint=False)

    fixed_doppler = rng.uniform(-doppler_uncertainty / 2, doppler_uncertainty / 2)

    noise_std = np.sqrt(sampling_rate / 2 / 10**(cno / 10))

    stream = []
    tone_labels = []   # optional, for debugging / evaluation

    for i in range(num_tones):
        
        # Pick random MFSK symbol
        symbol_idx = rng.integers(0, args.order)
        freq = frequencies[symbol_idx]

        # Doppler rate for this tone (varies per tone)
        dru = rng.uniform(-doppler_rate_uncertainty, doppler_rate_uncertainty)

        # Instantaneous frequency
        freq_t = freq + fixed_doppler + dru * t

        # Random phase
        phase = rng.uniform(0, 2*np.pi)

        # Unit-power sinusoid
        wave = np.sqrt(2) * np.sin(2*np.pi*freq_t * t + phase)

        # Add noise
        wave += rng.normal(0.0, noise_std, size=samples_per_tone)

        # Final normalization (same as training)
        wave /= np.sqrt(np.mean(wave**2))

        stream.append(wave)
        tone_labels.append(int(symbol_idx))

    stream = np.concatenate(stream)
    tone_labels = np.array(tone_labels)

    metadata = {
        "generator": os.path.basename(__file__),
        "generator_version": version,

        "sampling_rate": sampling_rate,
        "tone_duration": duration,

        "tones": frequencies,
        "noise": False,
        "num_tones": num_tones,

        "CNO_list": cno,
        "doppler_uncertainty_list": doppler_uncertainty,
        "doppler_rate_uncertainty": doppler_rate_uncertainty,
    }

    output = {
        "version": 1,
        "metadata": metadata,
        "stream": stream,
        "tone_labels": tone_labels
    }

    if args.output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_file = f"inference_tones_stream_{timestamp}.pkl"

    with open(f"{args.output_file}", "wb") as f:
        pkl.dump(output, f)


if __name__ == "__main__":
    main()
