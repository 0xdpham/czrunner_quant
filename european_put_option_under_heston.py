# European Put Option Pricing under the Heston Model (Monte Carlo)

import numpy as np
import matplotlib.pyplot as plt

SHOW_PLOTS = True        # set False to skip all plots
RUN_TASK_1_1 = True      # set False to skip the "new parameters" simulation


# ---------------------------------------------------------------------------
# Model functions
# NOTE: Heston_paths reads M, I, dt and rand from global scope, so those
# must be defined before it is called.
# ---------------------------------------------------------------------------
def SDE_vol(v0, kappa, theta, sigma, T, M, I, rand, row, cho_matrix):
    dt = T / M
    v = np.zeros((M + 1, I), dtype=float)
    v[0] = v0
    sdt = np.sqrt(dt)
    for t in range(1, M + 1):
        ran = np.dot(cho_matrix, rand[:, t])
        v[t] = np.maximum(
            0,
            v[t - 1] + kappa * (theta - v[t - 1]) * dt
            + np.sqrt(v[t - 1]) * sigma * ran[row] * sdt,
        )
    return v


def Heston_paths(S0, r, v, row, cho_matrix):
    S = np.zeros((M + 1, I), dtype=float)
    S[0] = S0
    sdt = np.sqrt(dt)
    for t in range(1, M + 1):
        ran = np.dot(cho_matrix, rand[:, t])
        S[t] = S[t - 1] * np.exp(
            (r - 0.5 * v[t - 1]) * dt + np.sqrt(v[t - 1]) * ran[row] * sdt
        )
    return S


def random_number_gen(M, I):
    return np.random.standard_normal((2, M + 1, I))


def plot_paths(S, V, n):
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(range(len(S)), S[:, :n])
    ax1.grid()
    ax1.set_title("Heston Price paths")
    ax1.set_ylabel("Price")
    ax1.set_xlabel("Timestep")

    ax2.plot(range(len(V)), V[:, :n])
    ax2.grid()
    ax2.set_title("Heston Volatility paths")
    ax2.set_ylabel("Volatility")
    ax2.set_xlabel("Timestep")


# ---------------------------------------------------------------------------
# Pricing functions
# ---------------------------------------------------------------------------
def heston_put_mc(S, K, r, T, t):
    """Monte Carlo price of a European put from simulated Heston paths S."""
    S_T = S[-1, :]                       # final prices
    payoff = np.maximum(K - S_T, 0)      # put payoff
    avg_payoff = np.mean(payoff)         # average over paths
    return np.exp(-r * (T - t)) * avg_payoff


def heston_call_mc(S, K, r, T, t):
    """Used here only for the put-call parity check."""
    payoff = np.maximum(S[-1, :] - K, 0)
    return np.exp(-r * (T - t)) * np.mean(payoff)


# ---------------------------------------------------------------------------
# Main simulation (original parameters)
# ---------------------------------------------------------------------------
v0 = 0.095
kappa_v = 5
sigma_v = 0.35
theta_v = 0.1
rho = 0.25

S0 = 40.53   # Current underlying asset price
r = 0.05     # Risk-free rate
M0 = 500     # Time steps per year
T = 12 / 12  # Years
M = int(M0 * T)
I = 20000    # Number of simulations
dt = T / M

np.random.seed(42)
rand = random_number_gen(M, I)

covariance_matrix = np.array([[1.0, rho],
                              [rho, 1.0]])
cho_matrix = np.linalg.cholesky(covariance_matrix)

V = SDE_vol(v0, kappa_v, theta_v, sigma_v, T, M, I, rand, 1, cho_matrix)
S = Heston_paths(S0, r, V, 0, cho_matrix)

K = 39
put_price = heston_put_mc(S, K, r, T, 0)
call_price = heston_call_mc(S, K, r, T, 0)

print("European Put Price under Heston: ", put_price)
print("Parity check (call - put):", call_price - put_price,
      "vs S0 - K*exp(-rT):", S0 - K * np.exp(-r * T))

if SHOW_PLOTS:
    plot_paths(S, V, 300)


# ---------------------------------------------------------------------------
# Task 1.1: new parameters (runs AFTER the pricing above, because it
# overwrites the globals I, rand and cho_matrix that Heston_paths uses)
# ---------------------------------------------------------------------------
if RUN_TASK_1_1:
    rho_new = -0.5
    kappa_v_new = 1.5
    I_new = 100

    I = I_new
    corr_mat_new = np.array([[1.0, rho_new],
                             [rho_new, 1.0]])
    cho_matrix = np.linalg.cholesky(corr_mat_new)
    rand = random_number_gen(M, I_new)

    V_new = SDE_vol(v0, kappa_v_new, theta_v, sigma_v, T, M, I_new, rand, 1, cho_matrix)
    S_new = Heston_paths(S0, r, V_new, 0, cho_matrix)

    print("European Put Price under Heston (new params, 100 paths): ",
          heston_put_mc(S_new, K, r, T, 0))

    if SHOW_PLOTS:
        plot_paths(S_new, V_new, 100)

if SHOW_PLOTS:
    plt.show()
