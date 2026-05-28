import numpy as np
import scipy.linalg
from controllers.pid_controller import PIDController

class MRACController:
    def __init__(self, gamma=5.0, am=3.0, bm=2.0):
        self.gamma = gamma
        
        self.Am = np.array([[0, 1], [-bm, -am]])
        self.Bm = np.array([[0], [bm]])
        
        Q = np.eye(2)
        self.P = scipy.linalg.solve_continuous_lyapunov(self.Am.T, -Q)
        
        self.B_plant = np.array([[0], [3.0]])
        
        base_pid = PIDController()
        self.K_x = np.array([[-base_pid.kp], [-base_pid.kd], [-base_pid.ki]]) 
        self.K_r = 0.0  
        
        self.W = np.zeros((4, 1))

        self.integral = 0.0
        
    def compute(self, x, xm, r, dt, t=0.0):
        
        x = x.reshape(2, 1)
        xm = xm.reshape(2, 1)
        e = x - xm  
        
        self.integral += x[0, 0] * dt
        
        x_aug = np.array([[x[0, 0]], [x[1, 0]], [self.integral]])
        
        phi = np.array([[np.sin(x[0, 0])], [1.0],[np.sin(2 * t)],[np.cos(2 * t)]])
        
        u_matrix = self.K_x.T @ x_aug + self.K_r * r - self.W.T @ phi
        u = float(u_matrix[0, 0])
        
        e_pb_matrix = e.T @ self.P @ self.B_plant
        e_pb = float(e_pb_matrix[0, 0])
        
        self.K_x += -self.gamma * x_aug * e_pb * dt
        self.K_r += -self.gamma * r * e_pb * dt
        self.W += self.gamma * phi * e_pb * dt
        
        return np.clip(u, -2.0, 2.0)