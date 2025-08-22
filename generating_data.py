import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm

# ------------------------
# Euler equations function
# ------------------------

def euler_eqs(t, omega, I, tau_func):
    omega = np.array(omega).flatten()
    tau = tau_func(omega)
    domega = np.linalg.solve(I, tau - np.cross(omega, I @ omega))
    return domega

# ------------------------
# Single case runner
# ------------------------

def run_single_case(
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz,
    COM, offset, mass_g,
    tspan, omega0
):
    if Ixx <= 0 or Iyy <= 0 or Izz <= 0:
        return None

    I = np.array([
        [ Ixx, -Ixy, -Ixz],
        [-Ixy,  Iyy, -Iyz],
        [-Ixz, -Iyz,  Izz]
    ])

    # Skip singular matrices
    det = np.linalg.det(I)
    if abs(det) < 1e-8:
        return None

    # Compute force vector (thrust) along +z
    g = 9.81
    thrust_magnitude = 1.1 * (mass_g) * g
    thrust_vector = np.array([0, 0, thrust_magnitude])

    # Compute moment arm (vector from COM to thrust point)
    r = offset - COM

    # Torque = r × F
    torque_vector = np.cross(r, thrust_vector)

    tau_func = lambda omega: torque_vector

    try:
        sol = solve_ivp(
            euler_eqs,
            tspan,
            omega0,
            args=(I, tau_func),
            method='RK23',
            rtol=1e-2,
            atol=1e-2
        )
        final_omega = sol.y[:, -1]
        omega_x, omega_y, omega_z = final_omega
        numerator = abs(omega_z)
        denominator = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        ratio = 0 if denominator == 0 else numerator / denominator

        # Round result to 2 decimal places
        ratio = round(ratio, 2)

    except Exception as e:
        omega_x, omega_y, omega_z = np.nan, np.nan, np.nan
        ratio = np.nan

    return [
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz,
        COM[0], COM[1], COM[2],
        offset[0], offset[1], offset[2],
        ratio
    ]

# ------------------------
# Parameter sweep runner
# ------------------------

def run_batch(
    ixx_val,
    iyy_vals, izz_vals,
    ixy_vals, ixz_vals, iyz_vals,
    COM_grid, offsets,
    mass_g,
    tspan, omega0,
    n_jobs=-1
):
    batch_params = list(product(
        [ixx_val],
        iyy_vals,
        izz_vals,
        ixy_vals,
        ixz_vals,
        iyz_vals,
        COM_grid,
        offsets
    ))

    parallel = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)

    results = parallel(
        delayed(run_single_case)(
            ixx, iyy, izz, ixy, ixz, iyz,
            COM, offset,
            mass_g,
            tspan, omega0
        )
        for (ixx, iyy, izz, ixy, ixz, iyz, COM, offset) in tqdm(batch_params, desc=f"Ixx={ixx_val}")
    )

    # Filter None results
    results = [r for r in results if r is not None]

    return results

# ------------------------
# Main execution
# ------------------------

if __name__ == "__main__":
    # Inertia parameter ranges
    ixx_vals = np.arange(50, 526, 100)
    iyy_vals = np.arange(50, 526, 100)
    izz_vals = np.arange(50, 526, 100)
    ixy_vals = np.arange(-50, 51, 20)
    ixz_vals = np.arange(-50, 51, 20)
    iyz_vals = np.arange(-50, 51, 20)

    # Define evenly spaced COM grid (cylinder)
    num_r = 3
    num_theta = 3
    num_z = 3

    r_max = 13.95
    z_min = 0
    z_max = 14

    r_vals = np.linspace(0, r_max, num_r)
    theta_vals = np.linspace(0, 2*np.pi, num_theta, endpoint=False)
    z_vals = np.linspace(z_min, z_max, num_z)

    COM_grid = []
    for r in r_vals:
        for theta in theta_vals:
            for z in z_vals:
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                COM_grid.append(np.array([x, y, z]))

    # Define 4 evenly spaced offsets along x
    offset_x_vals = np.linspace(0, 4, 4)
    offset_z = 11.5
    offsets = [np.array([x, 0, offset_z]) for x in offset_x_vals]

    omega0 = [0, 0, 90]
    tspan = (0, 10)
    mass_g = 3

    all_results = []

    for ixx in ixx_vals:
        batch_results = run_batch(
            ixx,
            iyy_vals,
            izz_vals,
            ixy_vals,
            ixz_vals,
            iyz_vals,
            COM_grid,
            offsets,
            mass_g,
            tspan,
            omega0,
            n_jobs=-1
        )
        all_results.extend(batch_results)

    df = pd.DataFrame(
        all_results,
        columns=[
            'Ixx','Iyy','Izz','Ixy','Ixz','Iyz',
            'COM_x','COM_y','COM_z',
            'Offset_x','Offset_y','Offset_z',
            'ratio'
        ]
    )
    df.to_csv('the_sweep.csv', index=False)

    print("Sweep complete. Data saved to sweep_inertia_COM_offset_results.csv.")
