import numpy as np

class SMCController:
    def __init__(self, lam=5.0, gamma=2.0, ks=1.0):
        self.lam = lam      # Sliding surface parameter
        self.gamma = gamma  # Adaptation rate for robust gain
        self.ks = ks        # Linear feedback gain on sliding surface
        
        self.K_hat = 0.0    # Adaptive robust gain
        
        # Pendulum parameters (m=1, l=1)
        self.g = 10.0
        self.b = 3.0        # Control multiplier (3 / ml^2)
        
    def compute(self, x, xm, x_ddot_m, dt):
       
        e = x[0] - xm[0]
        e_dot = x[1] - xm[1]
        
        # Sliding surface
        s = e_dot + self.lam * e
        
        # Plant dynamics f(x)
        fx = (3.0 * self.g / 2.0) * np.sin(x[0])
        
        # Equivalent control
        u_eq = (1.0 / self.b) * (-fx + x_ddot_m - self.lam * e_dot - self.ks * s)
        
        # Robust control
        u_rob = (1.0 / self.b) * (-self.K_hat * np.sign(s))
        
        u = u_eq + u_rob
        
        # Adaptation Law to bound the unknown disturbance
        self.K_hat += self.gamma * abs(s) * dt
        
        # Add a tiny decay to prevent gain drift
        self.K_hat -= 0.01 * self.K_hat * dt
        
        return np.clip(u, -2.0, 2.0)