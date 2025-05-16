import jax
import jax.numpy as jnp
from functools import partial
import os
import re


# Helper function to generate and save HLO to a file
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

        # Lower the jitted functions to get hlo text.
        lowered_computation = jitted_fn.lower(*example_args, **example_kwargs)

        # Get HLO text from the Lowered object
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



# Simple Linear Regression Prediction
def linear_regression_predict(params, X):
    w, b = params
    return jnp.dot(X, w) + b


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


# Linear Regression Update
def linear_regression_loss_for_grad(params, X, y):
    w_lg, b_lg = params
    predictions = jnp.dot(X, w_lg) + b_lg
    return jnp.mean((predictions - y) ** 2)


grad_lr_loss_fn = jax.grad(linear_regression_loss_for_grad)


def linear_regression_update_for_hlo(params, X, y, learning_rate):
    grads = grad_lr_loss_fn(params, X, y)
    w, b = params
    w_grad, b_grad = grads
    new_w = w - learning_rate * w_grad
    new_b = b - learning_rate * b_grad
    return (new_w, new_b)


# Euler Solver

def euler_solver_for_hlo(y0, a_decay_const, t_start, dt_fixed, num_steps_static):
    """
    Args:
        y0: Initial value (dynamic JAX array)
        a_decay_const: Decay constant (dynamic JAX array/scalar)
        t_start: Start time (can be dynamic, not used in this simple ODE body)
        dt_fixed: Time step (can be dynamic)
        num_steps_static: Number of steps (MUST BE STATIC for fori_loop)
    """

    def loop_body(i, val_y_current):
        y_next = val_y_current + dt_fixed * (-a_decay_const * val_y_current)
        return y_next

    final_y = jax.lax.fori_loop(0, num_steps_static, loop_body, y0)
    return final_y


# --- Make inputs and call save_hlo_to_file ---

key = jax.random.PRNGKey(42)
output_directory = "hlo_data"

# Linear Regression Predict
key_lr, subkey = jax.random.split(key)
X_lr_example = jax.random.normal(subkey, (5, 2))
params_lr_example = (jnp.array([0.5, -0.5]), jnp.array(1.0))
save_hlo_to_file("Linear Regression Predict",
                 linear_regression_predict,
                 example_args=(params_lr_example, X_lr_example),
                 output_dir=output_directory)

# Sigmoid
x_sigmoid_example = jnp.array([-1.0, 0.0, 1.0, 100.0, -100.0])
save_hlo_to_file("Sigmoid Function",
                 sigmoid,
                 example_args=(x_sigmoid_example,),
                 output_dir=output_directory)

# Linear Regression Update
y_lr_example = jnp.dot(X_lr_example, params_lr_example[0]) + params_lr_example[1] + 0.1
learning_rate_example = 0.01
save_hlo_to_file("Linear Regression Update (with grad)",
                 linear_regression_update_for_hlo,
                 example_args=(params_lr_example, X_lr_example, y_lr_example, learning_rate_example),
                 output_dir=output_directory)

# Euler Solver
y0_ode_example = jnp.array(1.0)
a_decay_example = 0.5
t_span_ode_example = (0.0, 1.0)  # (t_start_val, t_end_val)
dt_fixed_example = 0.1

t_start_val = t_span_ode_example[0]
t_end_val = t_span_ode_example[1]
# Cast num_steps to int for static argument
num_steps_py_int = int(round((t_end_val - t_start_val) / dt_fixed_example))


save_hlo_to_file("Euler Solver (simplified with static num_steps)",
                 euler_solver_for_hlo,
                 example_args=(y0_ode_example, a_decay_example, t_start_val, dt_fixed_example, num_steps_py_int),
                 static_argnums_for_jit=(4,),
                 output_dir=output_directory)

print(f"\nAll HLO generation attempts complete. Check the '{output_directory}' directory.")