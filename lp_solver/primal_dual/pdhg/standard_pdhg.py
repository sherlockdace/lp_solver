from typing import Optional

import chex
import jax.numpy as jnp
from jax import lax

from lp_solver._src.lp_state import StandardLPState


@chex.dataclass
class PDHGState(StandardLPState):
    vec_y: jnp.ndarray
    eta: float
    omega: float
    curr_iter: int
    max_iter: int
    tol: float


def standard_pdhg_solve(
    vec_c: jnp.ndarray,
    mat_A: jnp.ndarray,
    vec_b: jnp.ndarray,
    vec_x_init: Optional[jnp.ndarray] = None,
    vec_y_init: Optional[jnp.ndarray] = None,
    eta: Optional[float] = None,
    omega: Optional[float] = None,
    max_iter: int = 5000,
    tol: float = 1e-4,
    dtype: jnp.dtype = jnp.float32,
) -> PDHGState:
    r"""Solve the standardlinear programming problem using the PDHG algorithm.
    
    The standard linear programming problem is of the form:
    min c^T x
    s.t. Ax = b
    x >= 0
    
    The PDHG algorithm is a primal-dual method for solving this problem.
    The algorithm is described in many papers, for example:https://arxiv.org/abs/2106.04756
    
    Args:
        `vec_c`: The cost vector of the linear programming problem.
        `mat_A`: The constraint matrix of the linear programming problem.
        `vec_b`: The right-hand side vector of the linear programming problem.
        `vec_x_init`: The initial guess for the primal variable.
        `vec_y_init`: The initial guess for the dual variable.
        `eta`: The step size for the primal variable.
        `omega`: The step size for the dual variable.
        `max_iter`: The maximum number of iterations.
        `tol`: The tolerance for the stopping criterion.
        `dtype`: The data type of the variables.

    Returns:
        `PDHGState`: The final state of the PDHG algorithm.
    """
    init_state = init_state_genearte(
        vec_c=vec_c,
        mat_A=mat_A,
        vec_b=vec_b,
        vec_x_init=vec_x_init,
        vec_y_init=vec_y_init,
        eta=eta,
        omega=omega,
        dtype=dtype,
        max_iter=max_iter,
        tol=tol,
    )
    final_state = lax.while_loop(_cond_fn, _update_fn, init_state)
    return final_state


def init_state_genearte(
    vec_c: jnp.ndarray,
    mat_A: jnp.ndarray,
    vec_b: jnp.ndarray,
    vec_x_init: Optional[jnp.ndarray] = None,
    vec_y_init: Optional[jnp.ndarray] = None,
    eta: Optional[float] = None,
    omega: Optional[float] = None,
    dtype: jnp.dtype = jnp.float32,
    max_iter: int = 5000,
    tol: float = 1e-4,
) -> PDHGState:
    # Initialize primal and dual variables
    if vec_x_init is None:
        vec_x_init = jnp.zeros(mat_A.shape[1], dtype=dtype)
    if vec_y_init is None:
        vec_y_init = jnp.zeros(mat_A.shape[0], dtype=dtype)

    if eta is None:
        eta = 0.9 / (jnp.linalg.norm(mat_A, ord=2) + 1e-10)
    if omega is None:
        omega = 1.0

    # Initialize state
    state = PDHGState(
        vec_c=jnp.asarray(vec_c, dtype=dtype),
        mat_A=jnp.asarray(mat_A, dtype=dtype),
        vec_b=jnp.asarray(vec_b, dtype=dtype),
        vec_x=vec_x_init,
        vec_y=vec_y_init,
        eta=eta,
        omega=omega,
        curr_iter=0,
        max_iter=max_iter,
        tol=tol,
    )
    return state


def _cond_fn(pdhg_state: PDHGState) -> bool:
    """Condition for the PDHG solver to continue iterating."""
    logit1 = pdhg_state.curr_iter < pdhg_state.max_iter
    logit2 = jnp.linalg.norm(
        jnp.dot(pdhg_state.mat_A, pdhg_state.vec_x) - pdhg_state.vec_b
    ) > pdhg_state.tol * (1.0 + jnp.linalg.norm(pdhg_state.vec_b))
    lambd = pdhg_state.vec_c - jnp.dot(pdhg_state.mat_A.T, pdhg_state.vec_y)
    lambd = jnp.clip(lambd, None, 0)
    logit3 = jnp.linalg.norm(lambd) > pdhg_state.tol * (
        1.0 + jnp.linalg.norm(pdhg_state.vec_c)
    )
    logit4 = jnp.abs(
        jnp.vdot(pdhg_state.vec_b, pdhg_state.vec_y)
        - jnp.vdot(pdhg_state.vec_x, pdhg_state.vec_c)
    ) > pdhg_state.tol * (
        1.0
        + jnp.abs(jnp.vdot(pdhg_state.vec_b, pdhg_state.vec_y))
        + jnp.abs(jnp.vdot(pdhg_state.vec_x, pdhg_state.vec_c))
    )
    return jnp.logical_and(logit1, jnp.any(jnp.asarray([logit2, logit3, logit4])))


def _update_fn(pdhg_state: PDHGState) -> PDHGState:
    """Update the state of the PDHG solver."""
    tau = pdhg_state.eta / pdhg_state.omega
    sigma = pdhg_state.omega * pdhg_state.eta
    vec_x_new = pdhg_state.vec_x - tau * (
        pdhg_state.vec_c - jnp.dot(pdhg_state.mat_A.T, pdhg_state.vec_y)
    )
    vec_x_new = jnp.clip(vec_x_new, 0, None)
    tmp = 2 * vec_x_new - pdhg_state.vec_x
    vec_y_new = pdhg_state.vec_y + sigma * (
        pdhg_state.vec_b - jnp.dot(pdhg_state.mat_A, tmp)
    )

    new_state = PDHGState(
        vec_c=pdhg_state.vec_c,
        mat_A=pdhg_state.mat_A,
        vec_b=pdhg_state.vec_b,
        vec_x=vec_x_new,
        vec_y=vec_y_new,
        eta=pdhg_state.eta,
        omega=pdhg_state.omega,
        curr_iter=pdhg_state.curr_iter + 1,
        max_iter=pdhg_state.max_iter,
        tol=pdhg_state.tol,
    )
    return new_state
