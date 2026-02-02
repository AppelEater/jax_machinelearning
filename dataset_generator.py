import os
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

version = "1.0"

# -------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a tone sequence for inference testing"
    )

    # Signal parameters
    parser.add_argument("--sampling-rate", type=float, default=2000,
                        help="Sampling rate [Hz]")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Duration per tone [s]")
    parser.add_argument("--num-waveforms", type=int, default=1000,
                        help="Number of waveforms in each class")
    parser.add_argument("--order", type=int, default=8,
                        help="MFSK order")
    parser.add_argument("--fist-freq", type=int, default=150,
                        help="Frequency of the lowest tome")
    parser.add_argument("--freq-spacing", type=int, default=100,
                        help="Tones frequency spacing")
    parser.add_argument("--no-noise", action="store_true",
                        help="Do not include noise class")

    # Modulation / channel parameters
    parser.add_argument("--cno", nargs='*', type=float, default=[14.2],
                        help="Carrier-to-noise density C/N0 [dB-Hz]")
    parser.add_argument("--doppler-uncertainty", nargs='*', type=float, default=[90.0],
                        help="Max Doppler offset [Hz]")
    parser.add_argument("--doppler-rate-uncertainty", nargs='*', type=float, default=[16.67],
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
    num_waveforms = args.num_waveforms
    frequencies = [args.fist_freq + i * args.freq_spacing for i in range(args.order)]
    noise =  not(args.no_noise)
    CNO_list = args.cno
    doppler_uncertainty_list = args.doppler_uncertainty
    doppler_rate_uncertainty = args.doppler_rate_uncertainty

    if args.seed is None:
        key = jax.random.PRNGKey(time.time_ns())
    else:
        key = jax.random.PRNGKey(args.seed)

    samples = int(sampling_rate * duration)
    t = jnp.linspace(0, duration, samples)

    # -------------------------------------------------------------

    waveforms = []

    for CNO in CNO_list:
        noise_std = jnp.sqrt(sampling_rate / 2 / 10**(CNO / 10))
        for doppler_uncertainty in doppler_uncertainty_list:
            # Generate tones with noise
            for idx, freq in enumerate(frequencies):
                for i in range(num_waveforms):
                    key, subkey = jax.random.split(key)
                    
                    phase = jax.random.uniform(key, shape=(1,), minval=0, maxval=2*jnp.pi)
                    
                    freq_off = freq + jax.random.uniform(key, 1, minval=-doppler_uncertainty/2, maxval=doppler_uncertainty/2)
                    # Add Doppler rate uncertainty as a time-dependent offset (linear drift)
                    dru_effect = jax.random.uniform(subkey, shape=(1,), minval=-doppler_rate_uncertainty[0], maxval= doppler_rate_uncertainty[0]) * t
                    freq_off += dru_effect

                    wave = jnp.sqrt(2)*jnp.sin(2*jnp.pi*freq_off * t + phase)  ## Generate tone with power amplitude of 1 (sqrt(2) used for normalisation)
                    wave += jax.random.normal(subkey, shape=(samples)) * noise_std  ## Add noise
                    wave /= jnp.sqrt(jnp.mean(wave**2))
                    waveforms.append((wave, idx))  # Save wavefore incl. normalisation

            # Generate noise only class
            if noise:
                for i in range(num_waveforms):
                    key, subkey = jax.random.split(key)
                    wave = jax.random.normal(key, (samples))
                    waveforms.append((wave/jnp.sqrt(jnp.mean(wave**2)), 8))

    # -------------------------------------------------------------

    metadata = {
        "generator": os.path.basename(__file__),
        "generator_version": version,

        "sampling_rate": sampling_rate,
        "tone_duration": duration,

        "tones": frequencies,
        "noise": noise,
        "num_waveforms_per_class": num_waveforms,

        "CNO_list": CNO_list,
        "doppler_uncertainty_list": doppler_uncertainty_list,
        "doppler_rate_uncertainty": doppler_rate_uncertainty,
    }

    output = {
        "version": 2,
        "metadata": metadata,
        "waveforms": waveforms
    }

    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"training_waveforms_{timestamp}.pkl"

    with open(f"datasets/{args.output}", "wb") as f:
        pickle.dump(output, f)

    # -------------------------------------------------------------

    # Calculate mean power of all waveforms
    # mean_power = [jnp.mean(x[0]**2) for x in waveforms]
    # fig, ax = plt.subplots()
    # ax.plot(mean_power)
    # ax.set_xlabel("Waveforms")
    # ax.set_ylabel("Power")
    # fig.show()

    # Verify 100 random waveforms from the dataset by doing fft plot
    # for i in range(10):
    #     k = np.random.choice(len(waveforms))
    #     plt.figure()
    #     plt.plot(np.fft.fftfreq(len(waveforms[0][0]), 1/sampling_rate),jnp.log10(jnp.abs(jnp.fft.fft(waveforms[k][0]))**2/len(waveforms[0][0])))
    #     plt.title(f'Frequency: {frequnecies[waveforms[k][1]] if waveforms[k][1]!= 8 else "Noise"}')
    #     plt.show()
    # #    plt.close()



if __name__ == "__main__":
    main()
