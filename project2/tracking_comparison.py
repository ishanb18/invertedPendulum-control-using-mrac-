import matplotlib.pyplot as plt

from main import run
from controllers.pid_controller import PIDController
from controllers.mrac_controller import MRACController
from controllers.smc_controller import SMCController

def main():
    # Run PID
    print("Running PID...")
    t_pid, th_pid, _, pid_ise = run(PIDController(), "PID", render=False)
    
    # Run MRAC
    print("Running MRAC...")
    mrac = MRACController()
    t_mrac, th_mrac, _, mrac_ise = run(mrac, "MRAC", render=False)
    
    # Run SMC
    print("Running SMC...")
    smc = SMCController()
    t_smc, th_smc, _, smc_ise = run(smc, "SMC", render=False)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # The reference "path" is stabilization to 0
    plt.plot(t_pid, [0] * len(t_pid), 'k--', label='Reference Path (0 rad)', linewidth=2)
    
    plt.plot(t_pid, th_pid, label=f'PID (ISE: {pid_ise:.2f})', alpha=0.8)
    plt.plot(t_mrac, th_mrac, label=f'MRAC (ISE: {mrac_ise:.2f})', alpha=0.8)
    plt.plot(t_smc, th_smc, label=f'SMC (ISE: {smc_ise:.2f})', alpha=0.8)
    
    plt.title('Pendulum Stabilization Path Tracking (PID vs MRAC vs SMC)')
    plt.xlabel('Time (s)')
    plt.ylabel('Pendulum Angle (rad)')
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    output_filename = 'tracking_comparison_all.png'
    plt.savefig(output_filename, dpi=150)
    print(f"\nSaved tracking comparison graph to {output_filename}")
    plt.show()

if __name__ == "__main__":
    main()
