import numpy as np

class PIDController:
    def __init__(self, lam=0.333, m=1.0, l=1.0, g=10.0):
        self.a = (3 * g) / (2 * l)
        self.b = 3.0 / (m * l**2)
        
        p = np.sqrt(self.a)
        
        self.alpha = (lam**3)*self.a + 3*(lam**2)*p + 3*lam
        
        self.beta = 3*(lam**2) + (lam**3)*p
        
        self.kp = (self.alpha * p + 1) / (self.b * self.beta)
        self.ki = p / (self.b * self.beta)
        self.kd = self.alpha / (self.b * self.beta)
        self.tau_f = (lam**3) / self.beta
        
        self.int = 0
        self.prev_e = 0
        self.prev_d_filtered = 0

    def compute(self, theta, theta_dot, dt):
        e = -theta
        
        d_raw = (e - self.prev_e) / dt if dt > 0 else 0
        self.prev_e = e
        
        alpha_f = dt / (self.tau_f + dt)
        d_filtered = (1 - alpha_f) * self.prev_d_filtered + alpha_f * d_raw
        self.prev_d_filtered = d_filtered
        
        u_unclipped = self.kp*e + self.ki*self.int + self.kd*d_filtered
        
        u_clipped = np.clip(u_unclipped, -2.0, 2.0)
        

        Ka = 1.0 # Anti-windup gain
        difference = u_unclipped - u_clipped
        
        self.int += (e - Ka * difference) * dt

        return u_clipped