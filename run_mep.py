import numpy as np
import matplotlib.pyplot as plt

from mep_core import simulate_from_speed_sequence_kmh, MUSCLES

SPEEDS_PATH = "speeds.txt"   # relative to this script / repo root

def main():
    # 1) Load speeds from file (one speed per line, in km/h)
    speeds = np.loadtxt(SPEEDS_PATH, dtype=float)
    speeds = np.atleast_1d(speeds)   # make sure it's 1D

    print("Loaded speeds:", speeds)

    # 2) Generate time vector + activations using MEP
    t_all, act = simulate_from_speed_sequence_kmh(
        speeds_kmh=speeds,
        elevation=0.0,      # change if you start using elevation
        real_time=False
    )

    print("Time length:", len(t_all), "first/last:", t_all[0], t_all[-1])
    for m in MUSCLES:
        print(m, act[m].shape)

    # 3) Quick plot to check
    plt.figure()
    plt.plot(t_all, act['TA'],  label='TA')
    plt.plot(t_all, act['Sol'], label='Sol')
    plt.xlabel("time [s]")
    plt.ylabel("activation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
