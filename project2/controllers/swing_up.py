import numpy as np

class SwingUpController:
    def __init__(self, m=1.0, l=1.0):
        self.m = m
        self.l = l
        self.g = 10.0
        self.I = (1/3) * m * l**2

        self.E_des = m * self.g * l

    def compute(self, theta, theta_dot):
        E = 0.5 * self.I * theta_dot**2 + 0.5 * self.m * self.g * self.l * (1 + np.cos(theta))

        k = 2.0
        u = -k * (E - self.E_des) * theta_dot

        return np.clip(u, -2, 2)