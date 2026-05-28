<div align="center">

# 🔄 Nonlinear Adaptive Control of an Inverted Pendulum

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-OpenAI-412991?style=for-the-badge)](https://gymnasium.farama.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> **A rigorous, simulation-based comparison of PID, Model Reference Adaptive Control (MRAC), and Sliding Mode Control (SMC) on a nonlinear inverted pendulum under realistic disturbances and sensor noise.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Controllers](#-controllers)
  - [Energy-Based Swing-Up](#1-energy-based-swing-up)
  - [PID Controller](#2-pid-controller)
  - [MRAC Controller](#3-model-reference-adaptive-control-mrac)
  - [SMC Controller](#4-sliding-mode-control-smc)
- [Reference Model](#-reference-model)
- [Realistic Environment](#-realistic-simulation-environment)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Running Experiments](#-running-experiments)
- [Results & Analysis](#-results--analysis)
- [Mathematical Background](#-mathematical-background)
- [Dependencies](#-dependencies)

---

## 🎯 Overview

The **inverted pendulum** is one of the most studied benchmarks in control theory — it is inherently unstable, nonlinear, and sensitive to disturbances, making it an ideal testbed for advanced control strategies.

This project implements and rigorously compares **three control strategies** across a simulation environment enriched with:

- 🌊 **Continuous sinusoidal torque disturbances** (`0.5·sin(2t)`)
- 🎲 **Stochastic Gaussian torque noise** (`σ = 0.2 Nm`)
- 📡 **Sensor measurement noise** on both angle and angular velocity
- 🔻 **Low-pass filtering** on velocity measurements (realistic hardware emulation)
- ⚙️ **Parametric variations** — mass and length swept across 4 configurations

The key research question: *Which controller best balances transient performance, disturbance rejection, and robustness to plant parameter changes?*

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py / simulation runner              │
│                                                                 │
│   ┌───────────────┐    ┌──────────────────────────────────┐    │
│   │ ReferenceModel│    │       PendulumSimulation          │    │
│   │   (RK4 ODE)   │───▶│  (Gymnasium Pendulum-v1 wrapper)  │    │
│   └───────────────┘    │  + Disturbances + Sensor Noise    │    │
│           │            └──────────────┬───────────────────┘    │
│           │                           │                         │
│           ▼                           ▼                         │
│   ┌───────────────────────────────────────────────────────┐    │
│   │                  Mode Switcher (Hysteresis)            │    │
│   │    |θ| < 0.3 rad → Stabilization    (Controllers)     │    │
│   │    |θ| > 0.5 rad → Swing-Up         (Energy Pump)     │    │
│   └───────┬───────────────────┬──────────────┬────────────┘    │
│           │                   │              │                  │
│           ▼                   ▼              ▼                  │
│      SwingUp              PID / MRAC        SMC                 │
│     Controller           Controller      Controller             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎮 Controllers

### 1. Energy-Based Swing-Up

A **nonlinear energy-pumping** controller that injects energy into the pendulum until it reaches the upright equilibrium.

$$u = -k \cdot (E - E_{des}) \cdot \dot{\theta}$$

Where:
- $E = \frac{1}{2}I\dot{\theta}^2 + \frac{1}{2}mgl(1 + \cos\theta)$ — current total mechanical energy
- $E_{des} = mgl$ — target energy at the upright position
- $k = 2.0$ — pumping gain

Once $|\theta| < 0.3\,\text{rad}$, control is handed off to the stabilization controller via **hysteresis switching** (reverts if $|\theta| > 0.5\,\text{rad}$) to prevent chattering at the boundary.

---

### 2. PID Controller

An **analytically tuned** PID (not trial-and-error) with gains derived directly from the pendulum's physical parameters.

Given $a = \frac{3g}{2l}$, $b = \frac{3}{ml^2}$, and pole placement at $-\lambda$:

$$K_p = \frac{\alpha\sqrt{a} + 1}{b\beta}, \quad K_i = \frac{\sqrt{a}}{b\beta}, \quad K_d = \frac{\alpha}{b\beta}$$

**Key engineering features:**
- 🔇 **Filtered derivative** — low-pass filter with time constant $\tau_f = \lambda^3/\beta$ to suppress noise amplification
- 🛡️ **Anti-windup** — back-calculation prevents integrator wind-up during actuator saturation: `int += (e - Ka·(u_unclipped - u_clipped)) * dt`

---

### 3. Model Reference Adaptive Control (MRAC)

A **Direct MRAC** with online Lyapunov-based gradient adaptation. The plant is driven to track a stable reference model, with adaptive gains updated in real-time.

**Control law:**
$$u = K_x^T x_{aug} + K_r r - W^T \phi(x, t)$$

Where:
- $x_{aug} = [\theta,\, \dot{\theta},\, \int\theta\,dt]^T$ — augmented state (includes integrator for steady-state accuracy)
- $\phi(x,t) = [\sin\theta,\; 1,\; \sin(2t),\; \cos(2t)]^T$ — **nonlinear basis functions** to reject periodic disturbances
- $W$ — adaptive weight vector for nonlinear compensation

**Lyapunov-based adaptation laws** (gradient descent on $V = e^TPe$):

$$\dot{K}_x = -\gamma\, x_{aug}\, e^T P B_{plant}$$
$$\dot{K}_r = -\gamma\, r\, e^T P B_{plant}$$
$$\dot{W} = +\gamma\, \phi\, e^T P B_{plant}$$

The Lyapunov matrix $P$ is solved from the **continuous Lyapunov equation**:

$$A_m^T P + P A_m = -Q, \quad Q = I$$

This guarantees that the error dynamics are asymptotically stable when the plant is matched.

---

### 4. Sliding Mode Control (SMC)

An **adaptive robust SMC** that handles unknown disturbances via an online-adapted robust gain — preventing over-bounding and gain drift.

**Sliding surface:**
$$s = \dot{e} + \lambda e, \quad e = \theta - \theta_m$$

**Control law:**
$$u = \underbrace{\frac{1}{b}\left(-f(x) + \ddot{\theta}_m - \lambda\dot{e} - k_s s\right)}_{u_{eq}} + \underbrace{\frac{1}{b}\left(-\hat{K}\,\text{sgn}(s)\right)}_{u_{rob}}$$

Where $f(x) = \frac{3g}{2}\sin\theta$ captures the nonlinear plant dynamics.

**Adaptive gain law** with decay (prevents gain explosion):

$$\dot{\hat{K}} = \gamma |s| - 0.01\hat{K}$$

---

## 📐 Reference Model

A stable **2nd-order linear system** that defines the ideal desired closed-loop behavior:

$$\ddot{\theta}_m + a_m\dot{\theta}_m + b_m\theta_m = 0, \quad a_m = 3,\; b_m = 2$$

Integrated using **4th-order Runge-Kutta (RK4)** for numerical accuracy:

$$\theta_{m,k+1} = \theta_{m,k} + \frac{dt}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

During swing-up, the reference model is **synchronized to the actual state**, ensuring a smooth bumpless transfer when stabilization begins.

---

## 🌍 Realistic Simulation Environment

The `PendulumSimulation` class wraps Gymnasium's `Pendulum-v1` with several realism-enhancing features:

| Feature | Implementation |
|---|---|
| **Sinusoidal disturbance** | `0.5·sin(2t)` Nm added to every control input |
| **Stochastic torque noise** | `N(0, 0.2)` Nm Gaussian noise |
| **Angle measurement noise** | `N(0, 0.05)` rad |
| **Velocity measurement noise** | `N(0, 0.1)` rad/s |
| **Velocity low-pass filter** | `α = 0.3` first-order IIR filter |
| **Parametric variation** | Configurable mass `m` and length `l` |

---

## 📁 Project Structure

```
nonlinear_adaptive_control/
│
├── project2/
│   ├── main.py                          # Main benchmark runner & live visualizer
│   ├── simulation.py                    # Gymnasium wrapper with noise & disturbances
│   │
│   ├── controllers/
│   │   ├── pid_controller.py            # Analytically tuned PID + anti-windup
│   │   ├── mrac_controller.py           # Direct MRAC with Lyapunov adaptation
│   │   ├── smc_controller.py            # Adaptive Sliding Mode Controller
│   │   └── swing_up.py                  # Energy-based swing-up
│   │
│   ├── models/
│   │   └── reference_model.py           # RK4-integrated 2nd-order reference model
│   │
│   ├── trajectory_analysis.py           # MRAC across 4 plant configurations
│   ├── learning_rate_analysis.py        # γ sweep: convergence vs. stability
│   ├── learning_rate_trajectory_analysis.py  # Trajectory plots per learning rate
│   └── tracking_comparison.py          # Side-by-side controller tracking comparison
│
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.10+
```

### Installation

```bash
# Clone the repository
git clone https://github.com/ishanb18/invertedPendulum-control-using-mrac-.git
cd invertedPendulum-control-using-mrac-

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install numpy scipy matplotlib gymnasium pygame
```

---

## ▶️ Running Experiments

### 1. Full Benchmark (PID vs MRAC vs SMC)
```bash
cd project2
python main.py
```
Runs all three controllers, prints ISE scores, plots a comparison, then launches live rendered simulations for MRAC and SMC.

### 2. Trajectory Analysis (4 Plant Configurations)
```bash
python trajectory_analysis.py
```
Runs MRAC on nominal, heavy, light, and long-arm pendulum configurations. Plots angle, control effort, and mechanical energy for each.

### 3. Learning Rate Sensitivity Study
```bash
python learning_rate_trajectory_analysis.py
```
Sweeps adaptation gain `γ` and shows how convergence speed vs. oscillation trade-off changes.

### 4. Tracking Comparison Plot
```bash
python tracking_comparison.py
```
Generates a clean side-by-side tracking error comparison across all controllers.

---

## 📊 Results & Analysis

### Performance Metric: Integral Square Error (ISE)

$$\text{ISE} = \int_0^T \theta(t)^2 \, dt$$

Lower ISE = better tracking and disturbance rejection.

| Controller | Characteristics |
|---|---|
| **PID** | Fast initial response, fixed gains — degrades under parameter mismatch |
| **MRAC** | Online adaptation — maintains performance across plant variations |
| **SMC** | Robust against bounded disturbances — higher control effort, chattering mitigated by adaptive gain |

### Trajectory Analysis

The `trajectory_analysis.py` script tests MRAC across:

| Configuration | Mass | Length |
|---|---|---|
| Nominal | 1.0 kg | 1.0 m |
| Heavy | 3.0 kg | 1.0 m |
| Light | 0.5 kg | 1.0 m |
| Long Arm | 1.0 kg | 2.0 m |

MRAC's adaptive gains recalibrate online, maintaining stable tracking even when the true plant dynamics deviate significantly from the nominal design.

---

## 📐 Mathematical Background

### Pendulum Dynamics

$$\ddot{\theta} = \frac{3g}{2l}\sin\theta + \frac{3}{ml^2}u + d(t)$$

Where $d(t)$ represents external disturbances.

### Lyapunov Stability (MRAC)

Define the error $e = x - x_m$ and Lyapunov candidate:

$$V = e^T P e \geq 0$$

The adaptation laws are designed so that $\dot{V} \leq 0$, guaranteeing the tracking error converges to zero in the absence of disturbances and to a bounded set in their presence.

### Sliding Mode Stability

On the sliding surface $s = 0$, the error dynamics reduce to:

$$\dot{e} + \lambda e = 0 \implies e(t) = e(0)e^{-\lambda t}$$

The adaptive gain $\hat{K}$ ensures $s\dot{s} < 0$ (reaching condition) by bounding the unknown disturbance online.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical computation, matrix algebra |
| `scipy` | Lyapunov equation solver (`solve_continuous_lyapunov`) |
| `gymnasium` | `Pendulum-v1` physics simulation environment |
| `matplotlib` | Plotting and analysis |
| `pygame` | Live rendering of pendulum simulation |

---

<div align="center">

**Built with ❤️ for the love of control theory**

*If you found this useful, give it a ⭐ on GitHub!*

</div>
