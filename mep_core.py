# mep_core.py
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
from MEP_model import MEP_class

# Muscles in your MEP model
MUSCLES = ['TA', 'Sol', 'GasMed', 'GasLat']

# --------------------
# MEP INSTANCE SETUP
# --------------------

# Create ONE global instance so we don't re-import everything every call
_mep = MEP_class()
_mep.importRegressionCoef()
_mep.importBaseline()
_mep.importXP()
# If these are needed in your model, keep them, otherwise comment out:
try:
    _mep.importDelay()
except AttributeError:
    pass

try:
    _mep.importGain()
except AttributeError:
    pass


# --------------------
# CYCLE GENERATION
# --------------------

def generate_cycle_act(speed_kmh: float,
                       elevation: float = 0.0) -> Dict[str, np.ndarray]:
    """
    Generate one gait cycle of activations for the given speed (km/h).

    Uses your MEP_model.MEP_class.calc_MEP(elevation, speed).

    Returns
    -------
    cycle : dict
        {
          'TA':      np.ndarray [n_samples],
          'Sol':     np.ndarray [n_samples],
          'GasMed':  np.ndarray [n_samples],
          'GasLat':  np.ndarray [n_samples],
        }
    """
    signal = _mep.calc_MEP(elevation, speed_kmh)  # dict with keys TA, Sol, GasMed, GasLat

    cycle: Dict[str, np.ndarray] = {}
    for m in MUSCLES:
        cycle[m] = np.asarray(signal[m], dtype=float)

    return cycle


# --------------------
# TIMING FROM SPEED
# --------------------

def get_cycle_duration_from_speed(speed_kmh: float) -> float:
    """
    Approximate gait cycle duration using a simple cadence–speed relation.

    Parameters
    ----------
    speed_kmh : float
        Walking/running speed in km/h.

    Returns
    -------
    T_cycle : float
        Duration of one gait cycle [seconds].
    """
    if speed_kmh <= 0:
        raise ValueError("Speed must be > 0 km/h")

    # Example: walking: cadence grows roughly linearly with speed.
    # Adjust coefficients to your data if you have better relations.
    if speed_kmh <= 6.0:  # walking range
        cadence_spm = 90 + 8 * speed_kmh   # steps per minute
    else:  # running range
        cadence_spm = 150 + 3 * (speed_kmh - 6.0)

    steps_per_second   = cadence_spm / 60.0
    cycles_per_second  = steps_per_second / 2.0  # 2 steps = 1 gait cycle
    T_cycle            = 1.0 / cycles_per_second

    return T_cycle


# --------------------
# SEQUENCE OF SPEEDS
# --------------------

def simulate_from_speed_sequence_kmh(
    speeds_kmh: Sequence[float],
    elevation: float = 0.0,
    real_time: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    For each speed in speeds_kmh:
      1. Generate one gait cycle using generate_cycle_act().
      2. Assign a physical duration using get_cycle_duration_from_speed().
      3. Concatenate time and activations.

    Parameters
    ----------
    speeds_kmh : list/array of float
        Sequence of speeds in km/h (e.g. from file or IMU).
    elevation : float
        Elevation/grade input for the MEP model (if used).
    real_time : bool
        If True, sleeps so that each cycle takes approximately its
        physical duration in wall-clock time.

    Returns
    -------
    t_all : np.ndarray [N]
        Global time vector (seconds).
    activations : dict
        { muscle_name: np.ndarray [N] } with concatenated activations.
    """
    activations: Dict[str, List[float]] = {m: [] for m in MUSCLES}
    t_all: List[float] = []

    t_offset = 0.0  # global time (seconds)

    for speed_kmh in speeds_kmh:
        start_time = time.time()

        # 1) MEP for this speed
        cycle = generate_cycle_act(speed_kmh, elevation=elevation)
        n_samples = len(next(iter(cycle.values())))  # assume all muscles same length

        # 2) Duration of this gait cycle
        T_cycle = get_cycle_duration_from_speed(speed_kmh)

        # 3) Time stamps: first at t_offset, last at t_offset + T_cycle
        if n_samples > 1:
            dt = T_cycle / (n_samples - 1)
        else:
            dt = 0.0

        t_cycle = [t_offset + i * dt for i in range(n_samples)]
        t_all.extend(t_cycle)
        t_offset += T_cycle

        # 4) Concatenate activations
        for m in MUSCLES:
            activations[m].extend(cycle[m])

        # 5) Optional: simulate real time
        if real_time:
            elapsed = time.time() - start_time
            remaining = T_cycle - elapsed
            if remaining > 0:
                time.sleep(remaining)

    t_all_arr = np.asarray(t_all, dtype=float)
    act_arr   = {m: np.asarray(vals, dtype=float) for m, vals in activations.items()}
    return t_all_arr, act_arr


# --------------------
# OPTIONAL QUICK TEST
# --------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Example: three cycles at different speeds
    speeds_test = [3.6, 4.0, 4.5]
    t_all, act = simulate_from_speed_sequence_kmh(speeds_test, elevation=0.0, real_time=False)

    print("Time length:", len(t_all), "first/last:", t_all[0], t_all[-1])
    for m in MUSCLES:
        print(m, act[m].shape)

    plt.figure()
    plt.plot(t_all, act['TA'],  label='TA')
    plt.plot(t_all, act['Sol'], label='Sol')
    plt.xlabel("time [s]")
    plt.ylabel("activation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
