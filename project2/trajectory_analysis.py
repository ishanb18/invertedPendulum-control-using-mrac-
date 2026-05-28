import numpy as np
import matplotlib.pyplot as plt

from simulation import PendulumSimulation
from controllers.swing_up import SwingUpController
from controllers.mrac_controller import MRACController
from models.reference_model import ReferenceModel

def wrap(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def main():
    # Define test configurations (mass, length)
    configs = [
        {"m": 1.0, "l": 1.0, "title": "Nominal (m=1.0, l=1.0)"},
        {"m": 3.0, "l": 1.0, "title": "Heavy (m=3.0, l=1.0)"},
        {"m": 0.5, "l": 1.0, "title": "Light (m=0.5, l=1.0)"},
        {"m": 1.0, "l": 2.0, "title": "Long (m=1.0, l=2.0)"}
    ]
    
    # 3 Rows (Angle, Control, Energy) x 4 Columns (Configurations)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharex=True)
    g = 10.0
    
    print("Simulating Trajectories for 4 different configurations...")
    
    for col, config in enumerate(configs):
        m = config["m"]
        l = config["l"]
        title = config["title"]
        
        # Plant Physical Parameters
        I = (1/3) * m * l**2
        E_des = m * g * l  # Maximum potential energy at upright position
        
        # Initialize simulation and controllers
        sim = PendulumSimulation(render=False, m=m, l=l)
        swing = SwingUpController(m=m, l=l)
        mrac = MRACController()
        ref = ReferenceModel()
        
        theta, theta_dot = sim.reset()
        
        dt = 0.05
        T = []
        TH = []
        U = []
        E = []
        
        in_stabilization = False
        
        for i in range(750):  # Run for 37.5 seconds
            t = i * dt
            theta_w = wrap(theta)
            
            # Reference model update
            theta_m, theta_dot_m = ref.update(dt)
            
            # Mode switching logic (hysteresis)
            if not in_stabilization and abs(theta_w) < 0.3:
                in_stabilization = True
            elif in_stabilization and abs(theta_w) > 0.5:
                in_stabilization = False
                
            if not in_stabilization:
                u = swing.compute(theta, theta_dot)
                # Keep reference model in sync
                ref.theta_m = theta_w
                ref.theta_dot_m = theta_dot
            else:
                x = np.array([theta_w, theta_dot])
                xm = np.array([theta_m, theta_dot_m])
                u = mrac.compute(x, xm, r=0.0, dt=dt, t=t)
                
            # Calculate current Total Mechanical Energy
            curr_E = 0.5 * I * theta_dot**2 + 0.5 * m * g * l * (1 + np.cos(theta))
            
            # Store Data
            T.append(t)
            TH.append(theta_w)
            U.append(u)
            E.append(curr_E)
            
            # Step physics
            theta, theta_dot = sim.step(u, t)
            
        sim.close()
        
        # ==========================================
        # Plot Column Data
        # ==========================================
        
        # --- Panel 1: Angle (Top Row) ---
        ax1 = axes[0, col]
        ax1.plot(T, TH, color='tab:blue', label='Angle' if col==0 else "")
        ax1.axhline(0, color='red', linestyle='--', label='Target' if col==0 else "")
        ax1.set_title(title, fontweight='bold', fontsize=12)
        ax1.grid(True)
        if col == 0:
            ax1.set_ylabel('Angle (rad)', fontweight='bold')
            ax1.legend(loc='lower right')
            
        # --- Panel 2: Control Effort (Middle Row) ---
        ax2 = axes[1, col]
        ax2.plot(T, U, color='tab:green', label='Control' if col==0 else "")
        ax2.grid(True)
        if col == 0:
            ax2.set_ylabel('Torque (Nm)', fontweight='bold')
            ax2.legend(loc='lower right')
            
        # --- Panel 3: Energy (Bottom Row) ---
        ax3 = axes[2, col]
        ax3.plot(T, E, color='m', label='Energy' if col==0 else "")
        ax3.axhline(E_des, color='orange', linestyle='--', label='Target E' if col==0 else "")
        ax3.set_xlabel('Time (s)', fontweight='bold')
        ax3.grid(True)
        if col == 0:
            ax3.set_ylabel('Energy (J)', fontweight='bold')
            ax3.legend(loc='lower right')
            
    # Add a main title for the entire figure
    fig.suptitle('Pendulum Trajectory Analysis Across Parameter Variations', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    
    output_filename = 'trajectory_analysis_4_params.png'
    plt.savefig(output_filename, dpi=150)
    print(f"Graph saved successfully as {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()
