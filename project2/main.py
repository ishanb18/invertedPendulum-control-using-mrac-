import numpy as np
import matplotlib.pyplot as plt
import time

from simulation import PendulumSimulation
from controllers.swing_up import SwingUpController
from controllers.pid_controller import PIDController
from controllers.mrac_controller import MRACController
from controllers.smc_controller import SMCController
from models.reference_model import ReferenceModel

def wrap(theta):
    return (theta + np.pi) % (2*np.pi) - np.pi

def run(ctrl, name, render=False, m=1.0, l=1.0):
    sim = PendulumSimulation(render, m=m, l=l)
    swing = SwingUpController(m=m, l=l)
    ref = ReferenceModel()

    theta, theta_dot = sim.reset()

    dt = 0.05
    T, TH, U, ERR = [], [], [], []

    # Hysteresis for swing-up / stabilization mode switching
    # Enter stabilization when |θ| < 0.3, exit back to swing-up when |θ| > 0.5
    # This prevents rapid mode chattering near the boundary
    in_stabilization = False

    for i in range(700):
        t = i * dt

        theta_w = wrap(theta)
        
        # Update reference model (r=0 for stabilization) — now uses RK4
        theta_m, theta_dot_m = ref.update(dt)
        theta_ddot_m = -ref.am * theta_dot_m - ref.bm * theta_m # Acceleration of reference model

        # Hysteresis mode switching
        if not in_stabilization and abs(theta_w) < 0.3:
            in_stabilization = True
        elif in_stabilization and abs(theta_w) > 0.5:
            in_stabilization = False

        if not in_stabilization:
            u = swing.compute(theta, theta_dot)
            # Sync reference model to current state while swinging up
            ref.theta_m = theta_w
            ref.theta_dot_m = theta_dot
            mode = "Swing-Up"
        else:
            mode = f"Stabilization ({name})"
            if name == "PID":
                u = ctrl.compute(theta_w, theta_dot, dt)
            elif name == "MRAC":
                x = np.array([theta_w, theta_dot])
                xm = np.array([theta_m, theta_dot_m])
                u = ctrl.compute(x, xm, r=0.0, dt=dt, t=t)
            elif name == "SMC":
                x = np.array([theta_w, theta_dot])
                xm = np.array([theta_m, theta_dot_m])
                u = ctrl.compute(x, xm, theta_ddot_m, dt=dt)

        print(f"[{name}] Step {i:03d} | t={t:.2f}s | Mode: {mode:20s} | Theta: {theta_w:.3f}")

        theta, theta_dot = sim.step(u, t)

        T.append(t)
        TH.append(theta_w) # We track the wrapped angle for ISE
        U.append(u)
        ERR.append(theta_w**2)

        if render:
            time.sleep(0.02)
            if sim.check_quit():
                print("\n[User Interrupted] 'q' pressed. Closing simulation window...")
                break

    sim.close()

    ise = np.sum(ERR) * dt
    return T, TH, U, ise

def plot_motion(t, th, u):
    plt.figure(figsize=(10,6))

    plt.subplot(2,1,1)
    plt.plot(t, th)
    plt.title("Angle")

    plt.subplot(2,1,2)
    plt.plot(t, u)
    plt.title("Control")

    plt.tight_layout()
    plt.show()

def main():
    ctrls = {
        "PID": PIDController(),
        "MRAC": MRACController(),
        "SMC": SMCController()
    }

    results = {}
    t_ref = None

    print("\n=== RESULTS ===")

    for name, c in ctrls.items():
        t, th, u, ise = run(c, name)

        print(f"{name} ISE: {ise:.4f}")

        results[name] = th
        if t_ref is None:
            t_ref = t

        # Optional: uncomment to see individual plots
        # plot_motion(t, th, u)

    plt.figure(figsize=(10,6))
    for k, v in results.items():
        plt.plot(t_ref, v, label=k)

    plt.legend()
    plt.grid()
    plt.title("Controller Comparison (with Disturbance)")
    plt.xlabel("Time (s)")
    plt.ylabel("Theta (rad)")
    plt.show()

    # 🎬 LIVE SIMULATION
    print("\nRunning Live Simulation for MRAC...")
    run(MRACController(), "MRAC", render=True)
    
    print("\nRunning Live Simulation for SMC...")
    run(SMCController(), "SMC", render=True)

if __name__ == "__main__":
    main()