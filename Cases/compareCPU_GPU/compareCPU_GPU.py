#!/usr/bin/env python3
import h5py
import numpy as np

# ----------------------------------------------------------------------
# 1) FILES TO COMPARE
# ----------------------------------------------------------------------
FILE_GPU = "3d_GPU_turbulent_channel_flow_10.h5"
FILE_CPU = "3d_CPU_turbulent_channel_flow_10.h5"

# ----------------------------------------------------------------------
# 2) Dataset paths INSIDE the h5 file for each field.
# ----------------------------------------------------------------------
TEMPERATURE_PATH = "T"
DENSITY_PATH     = "rho"
VELOCITY_X_PATH  = "u"
VELOCITY_Y_PATH  = "v"
VELOCITY_Z_PATH  = "w"
TIME_PATH        = "Time"

# Tolerance used to decide "numerically identical" (accounts for floating-point
# round-off, e.g. from different summation order on GPU vs CPU). Set to 0.0
# to require bit-for-bit exact equality instead.
RTOL = 1e-10   # relative tolerance
ATOL = 1e-12   # absolute tolerance


def list_datasets(path):
    """Print every dataset found inside an h5 file (name + shape)."""
    print(f"\nDatasets found in '{path}':")
    with h5py.File(path, "r") as f:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}   shape={obj.shape}  dtype={obj.dtype}")
        f.visititems(show)


def load(path, dataset):
    """Load one dataset from one h5 file as a numpy array."""
    with h5py.File(path, "r") as f:
        if dataset not in f:
            return None
        return np.array(f[dataset])


def compare(name, a, b):
    """Print a simple comparison between two arrays for one field."""
    print(f"\n--- {name} ---")
    if a is None or b is None:
        print(f"  Could not find '{name}' in one or both files (check the path).")
        return False
    if a.shape != b.shape:
        print(f"  Shapes differ: GPU={a.shape}  CPU={b.shape} -- cannot compare directly.")
        return False

    a = a.astype(np.float64)
    b = b.astype(np.float64)
    diff = a - b

    exact_match = np.array_equal(a, b)
    close_match = np.allclose(a, b, rtol=RTOL, atol=ATOL)

    print(f"  GPU  ->  min={a.min():.6g}  max={a.max():.6g}  mean={a.mean():.6g}")
    print(f"  CPU  ->  min={b.min():.6g}  max={b.max():.6g}  mean={b.mean():.6g}")
    print(f"  Mean absolute difference : {np.abs(diff).mean():.6g}")
    print(f"  Max  absolute difference : {np.abs(diff).max():.6g}")
    print(f"  RMSE                     : {np.sqrt((diff ** 2).mean()):.6g}")

    if exact_match:
        print("  Verdict: EXACT MATCH (bit-for-bit identical)")
    elif close_match:
        print(f"  Verdict: MATCH within tolerance (rtol={RTOL}, atol={ATOL}), "
              f"but not bit-for-bit identical")
    else:
        n_diff = np.count_nonzero(~np.isclose(a, b, rtol=RTOL, atol=ATOL))
        pct_diff = 100.0 * n_diff / a.size
        print(f"  Verdict: DIFFERENT -- {n_diff} / {a.size} points "
              f"({pct_diff:.4g}%) exceed tolerance")

    return exact_match and close_match


def compare_scalar(name, a, b):
    """Print a comparison between two scalar (or 0-d/1-element) values, e.g. 'time'."""
    print(f"\n--- {name} ---")
    if a is None or b is None:
        print(f"  Could not find '{name}' in one or both files (check the path).")
        return False

    a_val = float(np.asarray(a).reshape(-1)[0])
    b_val = float(np.asarray(b).reshape(-1)[0])
    diff = a_val - b_val

    exact_match = a_val == b_val
    close_match = np.isclose(a_val, b_val, rtol=RTOL, atol=ATOL)

    print(f"  GPU  -> {a_val:.10g}")
    print(f"  CPU  -> {b_val:.10g}")
    print(f"  Absolute difference : {abs(diff):.6g}")

    if exact_match:
        print("  Verdict: EXACT MATCH (bit-for-bit identical)")
    elif close_match:
        print(f"  Verdict: MATCH within tolerance (rtol={RTOL}, atol={ATOL}), "
              f"but not bit-for-bit identical")
    else:
        print("  Verdict: DIFFERENT -- exceeds tolerance")

    return exact_match and close_match


def main():
    # list_datasets(FILE_GPU)

    results = {}

    time_gpu = load(FILE_GPU, TIME_PATH)
    time_cpu = load(FILE_CPU, TIME_PATH)
    results["Time"] = compare_scalar("Time", time_gpu, time_cpu)

    temp_gpu = load(FILE_GPU, TEMPERATURE_PATH)
    temp_cpu = load(FILE_CPU, TEMPERATURE_PATH)
    results["Temperature"] = compare("Temperature", temp_gpu, temp_cpu)

    dens_gpu = load(FILE_GPU, DENSITY_PATH)
    dens_cpu = load(FILE_CPU, DENSITY_PATH)
    results["Density"] = compare("Density", dens_gpu, dens_cpu)

    vx_gpu = load(FILE_GPU, VELOCITY_X_PATH)
    vy_gpu = load(FILE_GPU, VELOCITY_Y_PATH)
    vz_gpu = load(FILE_GPU, VELOCITY_Z_PATH)

    vx_cpu = load(FILE_CPU, VELOCITY_X_PATH)
    vy_cpu = load(FILE_CPU, VELOCITY_Y_PATH)
    vz_cpu = load(FILE_CPU, VELOCITY_Z_PATH)

    # Compare each velocity component individually -- two vector fields can
    # have the same magnitude but point in different directions, so checking
    # components is required to confirm an exact-same snapshot.
    results["Velocity u"] = compare("Velocity u", vx_gpu, vx_cpu)
    results["Velocity v"] = compare("Velocity v", vy_gpu, vy_cpu)
    results["Velocity w"] = compare("Velocity w", vz_gpu, vz_cpu)

    # ---- Overall verdict ----
    print("\n" + "=" * 50)
    print("OVERALL SUMMARY")
    print("=" * 50)
    for field, ok in results.items():
        print(f"  {field:<15}: {'MATCH' if ok else 'DIFFERENT / MISSING'}")

    if all(results.values()):
        print("\n>>> RESULT: The GPU and CPU snapshots are the SAME "
              f"(within rtol={RTOL}, atol={ATOL}).")
    else:
        print("\n>>> RESULT: The GPU and CPU snapshots are DIFFERENT.")


if __name__ == "__main__":
    main()
