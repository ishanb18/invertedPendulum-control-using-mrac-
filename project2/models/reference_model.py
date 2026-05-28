class ReferenceModel:
    def __init__(self, am=3.0, bm=2.0):
        self.am = am
        self.bm = bm

        self.theta_m = 0.0
        self.theta_dot_m = 0.0

    def _derivatives(self, theta_m, theta_dot_m):
        theta_ddot = -self.am * theta_dot_m - self.bm * theta_m
        return theta_dot_m, theta_ddot

    def update(self, dt):
        # Stage 1
        k1_pos, k1_vel = self._derivatives(self.theta_m, self.theta_dot_m)
        
        # Stage 2
        k2_pos, k2_vel = self._derivatives(
            self.theta_m + 0.5 * dt * k1_pos,
            self.theta_dot_m + 0.5 * dt * k1_vel
        )
        
        # Stage 3
        k3_pos, k3_vel = self._derivatives(
            self.theta_m + 0.5 * dt * k2_pos,
            self.theta_dot_m + 0.5 * dt * k2_vel
        )
        
        # Stage 4
        k4_pos, k4_vel = self._derivatives(
            self.theta_m + dt * k3_pos,
            self.theta_dot_m + dt * k3_vel
        )
        
        # Weighted average
        self.theta_m += (dt / 6.0) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
        self.theta_dot_m += (dt / 6.0) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)

        return self.theta_m, self.theta_dot_m