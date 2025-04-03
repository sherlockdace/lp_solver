from time import time

import jax 
import jax.numpy as jnp
import scipy
import scipy.optimize

from lp_solver.pdhg.standard_pdhg import PDHGSolver


jax.config.update("jax_enable_x64", True)

m = 50
n = 200
mat_A = jax.random.normal(jax.random.PRNGKey(1), (m, n))
vec_b = jax.random.normal(jax.random.PRNGKey(1), (m,))
vec_c = jax.random.normal(jax.random.PRNGKey(1), (n,)) / n + 100

solver = PDHGSolver(max_iter=50000, tol=1e-4, dtype=jnp.float64)

tic = time()
out = solver.solve(
    vec_c=vec_c,
    mat_A=mat_A,
    vec_b=vec_b,
)
toc = time()
print(f"Time taken: {toc - tic:.4f} seconds")
print(f"The optimal value is: {jnp.vdot(out.vec_c, out.vec_x)}")

tic = time()
out2 = scipy.optimize.linprog(
    c=vec_c,
    A_eq=mat_A,
    b_eq=vec_b,
    method="highs",
)
toc = time()
print(f"Time taken: {toc - tic:.4f} seconds")
print(f"The optimal value is: {out2.fun}")