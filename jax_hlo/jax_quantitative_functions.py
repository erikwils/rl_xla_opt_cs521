import jax
import jax.numpy as jnp
from jax import random
import jax.scipy.stats.norm as jax_norm
from functools import partial
import os
import re


def save_hlo_to_file(description, target_fn, example_args, example_kwargs=None,
                     static_argnums_for_jit=None, static_argnames_for_jit=None,
                     output_dir="hlo_output"):
    if example_kwargs is None:
        example_kwargs = {}
    base_filename = re.sub(r'\W+', '_', description.lower()).strip('_')
    if not base_filename:
        base_filename = "unnamed_function"
    hlo_filename = f"{base_filename}.hlo"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, hlo_filename)

    print(f"\n--- Generating HLO for: {description} (to be saved in {filepath}) ---")
    try:
        jitted_fn = jax.jit(target_fn,
                            static_argnums=static_argnums_for_jit,
                            static_argnames=static_argnames_for_jit)
        lowered_computation = jitted_fn.lower(*example_args, **example_kwargs)
        hlo_text = lowered_computation.as_text()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(hlo_text)
        print(f"Successfully saved HLO to: {filepath}")

    except Exception as e:
        print(f"Error generating or saving HLO for {description}: {e}")
        print("  Make sure example arguments match the function signature and types,")
        print("  and that static_argnums/static_argnames (if any) are correctly specified for JIT.")
        try:
            with open(filepath + ".error.txt", 'w', encoding='utf-8') as f_err:
                f_err.write(f"Error generating HLO for: {description}\n{str(e)}")
        except Exception as e_write:
            print(f"  Additionally, failed to write error to file: {e_write}")



# Global PRNG key
master_key = random.PRNGKey(0)


# Black-Scholes Option Call
def black_scholes_call(S0, K, T, r, sigma):
    d1 = (jnp.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    call = S0 * jax_norm.cdf(d1) - K * jnp.exp(-r * T) * jax_norm.cdf(d2)
    return call


# Geometric Brownian Motion (GBM) Path Simulation
def simulate_gbm_for_hlo(S0, mu, sigma, T, dt, N_steps, key):
    t = jnp.linspace(0, T, N_steps + 1)
    dW = random.normal(key, shape=(N_steps,)) * jnp.sqrt(dt)
    W_path = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dW)])
    S = S0 * jnp.exp((mu - 0.5 * sigma ** 2) * t + sigma * W_path)
    return S


# Monte Carlo Option Pricing
def monte_carlo_call_price_for_hlo(S0, K, T, r, sigma, num_sim_static, key):
    dt = T
    Z = random.normal(key, shape=(num_sim_static,))
    ST = S0 * jnp.exp((r - 0.5 * sigma ** 2) * dt + sigma * jnp.sqrt(dt) * Z)
    payoff = jnp.maximum(ST - K, 0)
    return jnp.exp(-r * T) * jnp.mean(payoff)


# Exponential Moving Average (EMA)
def exponential_moving_average(prices, alpha):
    # prices[0] is used as initial state, scan over price
    def ema_step(prev_ema, current_price):
        ema_val = alpha * current_price + (1 - alpha) * prev_ema
        return ema_val, ema_val  # carry, output

    # Handle empty or single element prices array
    if prices.shape[0] == 0:
        return jnp.array([])
    if prices.shape[0] == 1:
        return prices

    initial_ema = prices[0]
    _, ema_values_scan = jax.lax.scan(ema_step, initial_ema, prices[1:])
    return jnp.concatenate([jnp.array([initial_ema]), ema_values_scan])


# Sharpe Ratio
def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return jnp.mean(excess_returns) / (jnp.std(excess_returns) + 1e-9)


# CAPM Equation
def capm_expected_return(R_f, beta, market_return):
    return R_f + beta * (market_return - R_f)


# Value at Risk (VaR)
def value_at_risk(returns, alpha_quantile=0.05):
    return jnp.quantile(returns, alpha_quantile)


# Kelly Criterion
def kelly_criterion(mu, r, sigma):
    return (mu - r) / (sigma ** 2 + 1e-9)


# Mean-Variance Portfolio Optimization
def mean_variance_opt(mu_assets, Sigma_cov, lam_risk_aversion):
    inv_Sigma = jnp.linalg.inv(Sigma_cov)
    w = jnp.dot(inv_Sigma, mu_assets)
    w_sum = jnp.sum(w)
    w_normalized = jnp.where(jnp.abs(w_sum) < 1e-9, w,
                             w / (w_sum + 1e-9 * jnp.sign(w_sum) + 1e-12))
    return w_normalized


# Heston Model Simulation (Euler-Maruyama)
def heston_model_for_hlo(S0, v0, mu_heston, kappa, theta, sigma_vol, rho, T, dt, N_steps, key):
    keys = random.split(key, 2)

    Z1_all = random.normal(keys[0], shape=(N_steps,))
    Z2_all = random.normal(keys[1], shape=(N_steps,))

    dW1_all = Z1_all * jnp.sqrt(dt)
    dW2_all = (rho * Z1_all + jnp.sqrt(1 - rho ** 2) * Z2_all) * jnp.sqrt(dt)

    def heston_step(carry, inputs_dW):
        St_prev, vt_prev = carry
        dW1_curr, dW2_curr = inputs_dW

        vt_non_neg = jnp.maximum(vt_prev, 0)  # non-negative check before sqrt

        St_new = St_prev * jnp.exp((mu_heston - 0.5 * vt_non_neg) * dt + jnp.sqrt(vt_non_neg) * dW1_curr)
        vt_new = vt_prev + kappa * (theta - vt_prev) * dt + sigma_vol * jnp.sqrt(vt_non_neg) * dW2_curr

        vt_new = jnp.maximum(vt_new, 1e-8)  # Floor variance to small positive number

        return (St_new, vt_new), (St_new, vt_new)

    initial_carry = (S0, v0)
    dW_stacked = jnp.stack([dW1_all, dW2_all], axis=-1)

    _, (S_path_scan, v_path_scan) = jax.lax.scan(heston_step, initial_carry, dW_stacked)

    # Prepend initial values S0, v0 to the paths from scan
    S_path = jnp.concatenate([jnp.array([S0]), S_path_scan])
    v_path = jnp.concatenate([jnp.array([v0]), v_path_scan])

    return S_path, v_path


output_directory_qf = "hlo_data"

# Common parameters
S0_ex = jnp.array(100.0)
K_ex = jnp.array(100.0)
T_ex = jnp.array(1.0)  # Time to maturity in years
r_ex = jnp.array(0.05)  # Risk-free rate
sigma_ex = jnp.array(0.2)  # Volatility
mu_ex = jnp.array(0.1)  # Expected return for GBM/Kelly

# Black-Scholes
key_bs, master_key = random.split(master_key)
save_hlo_to_file("Black-Scholes Call Option",
                 black_scholes_call,
                 example_args=(S0_ex, K_ex, T_ex, r_ex, sigma_ex),
                 output_dir=output_directory_qf)

# 2. GBM Simulation
dt_ex = jnp.array(0.01)
N_steps_gbm_ex = int(T_ex / dt_ex)  # Static Python int
key_gbm, master_key = random.split(master_key)
save_hlo_to_file("Geometric Brownian Motion Path Simulation",
                 simulate_gbm_for_hlo,
                 example_args=(S0_ex, mu_ex, sigma_ex, T_ex, dt_ex, N_steps_gbm_ex, key_gbm),
                 static_argnums_for_jit=(5,),
                 output_dir=output_directory_qf)

# Monte Carlo Option Pricing
num_sim_static_ex = 1000
key_mc, master_key = random.split(master_key)
save_hlo_to_file("Monte Carlo Call Option Price",
                 monte_carlo_call_price_for_hlo,
                 example_args=(S0_ex, K_ex, T_ex, r_ex, sigma_ex, num_sim_static_ex, key_mc),
                 static_argnums_for_jit=(5,),
                 output_dir=output_directory_qf)

# Exponential Moving Average
prices_ex = jnp.array([10.0, 10.2, 10.1, 10.5, 10.3, 10.6])
alpha_ema_ex = jnp.array(0.1)
key_ema, master_key = random.split(master_key)
save_hlo_to_file("Exponential Moving Average",
                 exponential_moving_average,
                 example_args=(prices_ex, alpha_ema_ex),
                 output_dir=output_directory_qf)

# Sharpe Ratio
returns_ex = random.normal(key_ema, (100,)) * 0.01 + 0.0005  # Example returns
risk_free_rate_ex = jnp.array(0.02 / 252)  # Daily risk-free
key_sr, master_key = random.split(master_key)
save_hlo_to_file("Sharpe Ratio",
                 sharpe_ratio,
                 example_args=(returns_ex, risk_free_rate_ex),
                 output_dir=output_directory_qf)

# CAPM Equation
Rf_capm_ex = jnp.array(0.02)
beta_capm_ex = jnp.array(1.2)
market_return_capm_ex = jnp.array(0.08)
key_capm, master_key = random.split(master_key)
save_hlo_to_file("CAPM Expected Return",
                 capm_expected_return,
                 example_args=(Rf_capm_ex, beta_capm_ex, market_return_capm_ex),
                 output_dir=output_directory_qf)

# Value at Risk (VaR)
alpha_var_ex = jnp.array(0.05)  # 5% VaR
key_var, master_key = random.split(master_key)
save_hlo_to_file("Value at Risk (VaR)",
                 value_at_risk,
                 example_args=(returns_ex, alpha_var_ex),
                 output_dir=output_directory_qf)

# Kelly Criterion
r_kelly_ex = r_ex
mu_kelly_ex = mu_ex
sigma_kelly_ex = sigma_ex
key_kelly, master_key = random.split(master_key)
save_hlo_to_file("Kelly Criterion",
                 kelly_criterion,
                 example_args=(mu_kelly_ex, r_kelly_ex, sigma_kelly_ex),
                 output_dir=output_directory_qf)

# Mean-Variance Portfolio Optimization
num_assets_ex = 3
key_mv, subkey_mu, subkey_sigma = random.split(master_key, 3)
mu_assets_ex = random.uniform(subkey_mu, (num_assets_ex,), minval=0.01, maxval=0.15)
# Create a random positive definite covariance matrix
_A = random.normal(subkey_sigma, (num_assets_ex, num_assets_ex))
Sigma_cov_ex = jnp.dot(_A, _A.T) * 0.01 + jnp.eye(num_assets_ex) * 0.001  # Ensure positive definite
lam_risk_aversion_ex = jnp.array(0.5)
master_key = key_mv
save_hlo_to_file("Mean-Variance Portfolio Optimization",
                 mean_variance_opt,
                 example_args=(mu_assets_ex, Sigma_cov_ex, lam_risk_aversion_ex),
                 output_dir=output_directory_qf)

# Heston Model Simulation
v0_heston_ex = jnp.array(sigma_ex ** 2)
mu_heston_ex = mu_ex
kappa_heston_ex = jnp.array(2.0)  # Mean-reversion speed for variance
theta_heston_ex = jnp.array(sigma_ex ** 2)  # Long-term mean of variance
sigma_vol_heston_ex = jnp.array(0.1)
rho_heston_ex = jnp.array(-0.7)  # Correlation S and v
N_steps_heston_ex = int(T_ex / dt_ex)
key_heston, master_key = random.split(master_key)
save_hlo_to_file("Heston Model Simulation",
                 heston_model_for_hlo,
                 example_args=(S0_ex, v0_heston_ex, mu_heston_ex, kappa_heston_ex, theta_heston_ex,
                               sigma_vol_heston_ex, rho_heston_ex, T_ex, dt_ex, N_steps_heston_ex, key_heston),
                 static_argnums_for_jit=(9,),
                 output_dir=output_directory_qf)

print(f"\nAll Quantitative Finance HLO generation attempts complete. Check the '{output_directory_qf}' directory.")