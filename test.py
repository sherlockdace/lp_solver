from time import time

import jax
import jax.numpy as jnp
import scipy
import scipy.optimize

from lp_solver import (
    standard_pdhg_solve,
    standard_extragradient_solve,
    eag_c_solve,
    g_eag_solve,
    aeg_solve,
)

jax.config.update("jax_enable_x64", True)

m = 100
n = 500
mat_A = jax.random.normal(jax.random.PRNGKey(1), (m, n))
vec_b = jax.random.normal(jax.random.PRNGKey(1), (m,))
vec_c = jax.random.normal(jax.random.PRNGKey(1), (n,)) / n + 5

tic = time()
state = standard_pdhg_solve(
    vec_c=vec_c,
    mat_A=mat_A,
    vec_b=vec_b,
    dtype=jnp.float64,
    tol=1e-8,
    max_iter=50000,
)
toc = time()
print(
    f"PDHG: Time taken: {toc - tic:.4f} seconds, current iter: {state.curr_iter}",
    end="\t",
)
print(f"The optimal value is: {jnp.vdot(state.vec_c, state.vec_x)}")

tic = time()
state = standard_extragradient_solve(
    vec_c=vec_c,
    mat_A=mat_A,
    vec_b=vec_b,
    dtype=jnp.float64,
            tol=1e-8,
    max_iter=50000,
)
toc = time()
print(
    f"EG: Time taken: {toc - tic:.4f} seconds, current iter: {state.curr_iter}",
    end="\t",
)
print(f"The optimal value is: {jnp.vdot(state.vec_c, state.vec_x)}")

tic = time()
state = eag_c_solve(
    vec_c=vec_c,
    mat_A=mat_A,
    vec_b=vec_b,
    dtype=jnp.float64,
    tol=1e-8,
    max_iter=50000,
)
toc = time()
print(
    f"EAGC: Time taken: {toc - tic:.4f} seconds, current iter: {state.curr_iter}",
    end="\t",
)
print(f"The optimal value is: {jnp.vdot(state.vec_c, state.vec_x)}")

tic = time()
state = g_eag_solve(
    vec_c=vec_c,
    mat_A=mat_A,
    vec_b=vec_b,
    dtype=jnp.float64,
    tol=1e-8,
    max_iter=50000,
)
toc = time()
print(
    f"G_EAG: Time taken: {toc - tic:.4f} seconds, current iter: {state.curr_iter}",
    end="\t",
)
print(f"The optimal value is: {jnp.vdot(state.vec_c, state.vec_x)}")

tic = time()
state = aeg_solve(
    vec_c=vec_c,
    mat_A=mat_A,
    vec_b=vec_b,
    dtype=jnp.float64,
    tol=1e-8,
    max_iter=50000,
)
toc = time()
print(
    f"AEG: Time taken: {toc - tic:.4f} seconds, current iter: {state.curr_iter}",
    end="\t",
)
print(f"The optimal value is: {jnp.vdot(state.vec_c, state.vec_x_x)}")

tic = time()
out2 = scipy.optimize.linprog(
    c=vec_c,
    A_eq=mat_A,
    b_eq=vec_b,
    method="highs",
)
toc = time()
print(f"SCIPY: Time taken: {toc - tic:.4f} seconds", end="\t")
print(f"The optimal value is: {out2.fun}")
