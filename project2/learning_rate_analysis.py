import numpy as np
import matplotlib.pyplot as plt

from main import run
from controllers.pid_controller import PIDController
from controllers.mrac_controller import MRACController

def main():
    learning_rates = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0]
    mrac_ise_values = []
    
    print("Running PID baseline...")
    _, _, _, pid_ise = run(PIDController(), "PID", render=False)
    print(f"PID Baseline ISE: {pid_ise:.4f}")
    
    print("\nRunning MRAC for different learning rates...")
    for lr in learning_rates:
        mrac = MRACController(gamma=lr)
        _, _, _, ise = run(mrac, "MRAC", render=False)
        mrac_ise_values.append(ise)
        print(f"MRAC (gamma={lr}) ISE: {ise:.4f}")
        
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, mrac_ise_values, marker='o', linestyle='-', color='b', label='MRAC')
    plt.axhline(y=pid_ise, color='r', linestyle='--', label='PID Baseline')
    
    plt.title('Integral Square Error (ISE) vs. MRAC Learning Rate')
    plt.xlabel('Learning Rate (gamma)')
    plt.ylabel('Integral Square Error (ISE)')
    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    # Save the plot
    plt.savefig('learning_rate_analysis.png')
    print("\nPlot saved as learning_rate_analysis.png")
    plt.show()

if __name__ == "__main__":
    main()
