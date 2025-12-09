import os
import subprocess
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mep_core import simulate_from_speed_sequence_kmh, MUSCLES

# =========================
# CONFIG: PATHS & FILES
# =========================

# Root of your CEINMS-RT repo on the Pi
CEINMS_REPO = "/home/pi/ceinms-rt"   # <-- change if different

# Execution config & subject (relative to CEINMS_REPO)
EXEC_XML    = os.path.join("cfg", "LowerLimbModel", "executionRT.xml")
SUBJECT_XML = os.path.join("cfg", "LowerLimbModel", "data", "calibrated_model_sub06_walking.xml")

# Trial folder (where emgFilt.sto, ik.sto, id.sto live)
TRIAL_FOLDER_REL  = os.path.join("cfg", "LowerLimbModel", "data", "pi_trial")
RESULT_FOLDER_REL = os.path.join(TRIAL_FOLDER_REL, "results")

TRIAL_FOLDER  = os.path.join(CEINMS_REPO, TRIAL_FOLDER_REL)
RESULT_FOLDER = os.path.join(CEINMS_REPO, RESULT_FOLDER_REL)

EXCITATIONS_FILE = os.path.join(TRIAL_FOLDER, "emgFilt.sto")
IK_FILE          = os.path.join(TRIAL_FOLDER, "ik.sto")
ID_FILE          = os.path.join(TRIAL_FOLDER, "id.sto")
TORQUE_FILE      = os.path.join(RESULT_FOLDER, "Torque.sto")  # <-- adjust name if needed

# CEINMS executable on the Pi (adjust if name/path differ)
CEINMS_EXE = os.path.join(CEINMS_REPO, "bin", "CEINMS")


# =========================
# SAVE .sto HELPERS
# =========================

def save_excitations_sto(time_vec: np.ndarray,
                         activations: Dict[str, np.ndarray],
                         filename: str) -> None:
    """
    Save CEINMS excitations in "Normalized EMG Linear Envelopes" format:

    Normalized EMG Linear Envelopes
    nRows=...
    nColumns=...
    endheader
    time   tib_ant_r   soleus_r   lat_gas_r   med_gas_r
    ...
    """
    # Map MEP muscles -> CEINMS EMG names
    name_map = {
        "TA":     "tib_ant_r",
        "Sol":    "soleus_r",
        "GasLat": "lat_gas_r",
        "GasMed": "med_gas_r",
    }

    data = {"time": time_vec}
    for internal_name, file_name in name_map.items():
        data[file_name] = activations[internal_name]

    col_order = ["time", "tib_ant_r", "soleus_r", "lat_gas_r", "med_gas_r"]
    df = pd.DataFrame(data)[col_order]

    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(filename, "w") as f:
        f.write("Normalized EMG Linear Envelopes\n")
        f.write(f"nRows={df.shape[0]}\n")
        f.write(f"nColumns={df.shape[1]}\n")
        f.write("endheader\n")
        df.to_csv(f, sep="\t", index=False, float_format="%.6f")


def save_zero_ik_sto(time_vec: np.ndarray,
                     dof_names: Sequence[str],
                     filename: str) -> None:
    """
    Create IK file with all joint angles = 0.

    Header format:
        version=1
        nRows=...
        nColumns=...
        inDegrees= yes
        Units are S.I. units (second, meters, Newtons, ...)
        Angles are in degrees.
        endheader
        time  <dof1>  <dof2> ...
    """
    data = {"time": time_vec}
    for dof in dof_names:
        data[dof] = np.zeros_like(time_vec)
    df = pd.DataFrame(data)

    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(filename, "w") as f:
        f.write("version=1\n")
        f.write(f"nRows={df.shape[0]}\n")
        f.write(f"nColumns={df.shape[1]}\n")
        f.write("inDegrees= yes\n")
        f.write("Units are S.I. units (second, meters, Newtons, ...)\n")
        f.write("Angles are in degrees.\n")
        f.write("endheader\n")
        df.to_csv(f, sep="\t", index=False, float_format="%.6f")


def save_zero_id_sto(time_vec: np.ndarray,
                     moment_names: Sequence[str],
                     filename: str) -> None:
    """
    Create ID file with all joint moments = 0.

    Header format similar to existing ID .sto in your dataset.
    """
    data = {"time": time_vec}
    for m_name in moment_names:
        data[m_name] = np.zeros_like(time_vec)
    df = pd.DataFrame(data)

    folder = os.path.dirname(filename)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(filename, "w") as f:
        f.write("version=1\n")
        f.write(f"nRows={df.shape[0]}\n")
        f.write(f"nColumns={df.shape[1]}\n")
        f.write("inDegrees= no\n")
        f.write("Units are S.I. units (second, Newton-meters, ...)\n")
        f.write("endheader\n")
        df.to_csv(f, sep="\t", index=False, float_format="%.6f")


# =========================
# CEINMS RUN & TORQUE LOADING
# =========================

def run_ceinms() -> subprocess.CompletedProcess:
    """
    Run CEINMS-RT on the Pi using the configured XMLs and trial folder.
    """
    # Sanity checks
    exec_xml_full    = os.path.join(CEINMS_REPO, EXEC_XML)
    subject_xml_full = os.path.join(CEINMS_REPO, SUBJECT_XML)

    for label, path in [("CEINMS exe", CEINMS_EXE),
                        ("executionRT.xml", exec_xml_full),
                        ("subject xml", subject_xml_full)]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[PATH ERROR] {label} not found at:\n  {path}")

    print("[INFO] Running CEINMS-RT on Pi...")
    print("  cwd:", CEINMS_REPO)
    print("  exe:", CEINMS_EXE)

    cmd = [
        CEINMS_EXE,
        "-e", EXEC_XML,
        "-s", SUBJECT_XML,
        "-r", RESULT_FOLDER_REL,
        "-p", TRIAL_FOLDER_REL,
        "-g",
    ]

    result = subprocess.run(
        cmd,
        cwd=CEINMS_REPO,
        capture_output=True,
        text=True
    )

    print("[INFO] CEINMS return code:", result.returncode)
    print("----- STDOUT -----")
    print(result.stdout)
    print("----- STDERR -----")
    print(result.stderr)

    if result.returncode != 0:
        print("[ERROR] CEINMS exited with non-zero code.")
    return result


def load_torque_sto(filename: str) -> pd.DataFrame:
    """
    Load a CEINMS/OpenSim .sto torque file into a DataFrame.
    Assumes header ends with 'endheader'.
    """
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("No 'endheader' found in torque file.")
            if line.strip().lower() == "endheader":
                break
        df = pd.read_csv(f, sep=r"\s+", engine="python")
    return df


def plot_torques(torques_df: pd.DataFrame,
                 cols: Sequence[str] | None = None,
                 title: str = "CEINMS torques") -> None:
    time = torques_df["time"]
    if cols is None:
        cols = [c for c in torques_df.columns if c != "time"]

    plt.figure(figsize=(8, 4))
    for c in cols:
        plt.plot(time, torques_df[c], label=c)
    plt.xlabel("time [s]")
    plt.ylabel("torque [Nm]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================
# FULL PIPELINE
# =========================

def build_trial_from_speeds(speeds_kmh: Sequence[float]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    1) MEP core: speeds -> time + activations
    2) Write emgFilt.sto, ik.sto, id.sto with matching length/time.
    """
    # 1) MEP → activations
    t_all, activations = simulate_from_speed_sequence_kmh(
        speeds_kmh=speeds_kmh,
        elevation=0.0,
        real_time=False
    )

    # 2) Save excitations
    save_excitations_sto(t_all, activations, EXCITATIONS_FILE)

    # 3) IK: all zero angles
    # TODO: adapt this list to the DOFs your executionIK XML expects
    dof_names = ["ankle_angle_r"]
    save_zero_ik_sto(t_all, dof_names, IK_FILE)

    # 4) ID: all zero torques
    # TODO: adapt this list to the moment names in your ID file / plugin config
    moment_names = [
        "ankle_moment_r",
    ]
    save_zero_id_sto(t_all, moment_names, ID_FILE)

    return t_all, activations


def run_full_pipeline_from_speeds_file(speeds_file: str) -> None:
    # Load speeds from file (one speed per line, in km/h)
    speeds = np.loadtxt(speeds_file, dtype=float)
    speeds = np.atleast_1d(speeds)

    print("Loaded speeds:", speeds)

    # Build trial data
    t_all, activations = build_trial_from_speeds(speeds)

    # Run CEINMS
    result = run_ceinms()
    if result.returncode != 0:
        print("[PIPELINE] CEINMS failed; not loading torque.")
        return

    # Load torques
    if not os.path.isfile(TORQUE_FILE):
        print("[PIPELINE] Torque file not found at:", TORQUE_FILE)
        return

    torques_df = load_torque_sto(TORQUE_FILE)
    print(torques_df.head())
    plot_torques(torques_df)


if __name__ == "__main__":
    # Example: speeds file in same folder as this script
    SPEEDS_FILE = "speeds.txt"
    run_full_pipeline_from_speeds_file(SPEEDS_FILE)
