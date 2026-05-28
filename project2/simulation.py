import gymnasium as gym
import numpy as np
import math

class PendulumSimulation:
    def __init__(self, render=False, m=1.0, l=1.0):
        self.env = gym.make("Pendulum-v1",
                            render_mode="human" if render else None)
        
        self.env.unwrapped.m = m
        self.env.unwrapped.l = l

        self.alpha = 0.3  
        self.theta_dot_filtered = 0.0
        self.filter_initialized = False

    def reset(self, init=[np.pi, 0]):
        self.env.reset()
        self.env.unwrapped.state = np.array(init)
        # Initialize filter state to the initial velocity
        self.theta_dot_filtered = init[1]
        self.filter_initialized = True
        return init

    def step(self, u, t=0.0):
        # Inject continuous sine disturbance + random torque disturbance
        dist = 0.5 * np.sin(2 * t) + np.random.normal(0, 0.2)
        
        # We apply the control + disturbance, but the environment clips the TOTAL torque to [-2, 2]
        obs, _, _, _, _ = self.env.step([np.clip(u + dist, -2, 2)])
        c, s, d = obs
        
        # True state
        theta_true = math.atan2(s, c)
        theta_dot_true = d
        
        # Inject Measurement Noise (Sensors are rarely perfect)
        theta_meas = theta_true + np.random.normal(0, 0.05)
        theta_dot_meas = theta_dot_true + np.random.normal(0, 0.1)
        
        # Low-pass filter on theta_dot to reduce noise amplification in controllers
        self.theta_dot_filtered = (self.alpha * self.theta_dot_filtered +
                                   (1 - self.alpha) * theta_dot_meas)
        
        return theta_meas, self.theta_dot_filtered

    def check_quit(self):
        """Allows user to close the pygame window gracefully by pressing 'q' or the close button."""
        if self.env.render_mode == "human":
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    return True
        return False

    def close(self):
        self.env.close()