from typing import Optional

import chex
import jax.numpy as jnp
from jax import lax

from lp_solver import StandardLPState


@chex.dataclass
class EGState(StandardLPState):
    vec_y: jnp.ndarray
    alpha: float
    curr_iter: int
    max_iter: int
    tol: float


def standard_extragradient_solve(
    vec_c: jnp.ndarray,
    mat_A: jnp.ndarray,
    vec_b: jnp.ndarray,
    vec_x_init: Optional[jnp.ndarray] = None,
    vec_y_init: Optional[jnp.ndarray] = None,
    alpha: Optional[float] = None,
    max_iter: int = 5000,
    tol: float = 1e-4,
    dtype: jnp.dtype = jnp.float32,
) -> EGState:
    r"""Solve the standard linear programming problem using the extragradient algorithm.

    The standard linear programming problem is of the form:
    min c^T x
    s.t. Ax = b
    x >= 0

    The extragradient algorithm is a primal-dual method for solving this problem.
    See like https://arxiv.org/abs/2102.07922 for more details.

    Args:
        `vec_c`: The cost vector of the linear programming problem.
        `mat_A`: The constraint matrix of the linear programming problem.
        `vec_b`: The right-hand side vector of the linear programming problem.
        `vec_x_init`: The initial guess for the solution.
        `vec_y_init`: The initial guess for the dual solution.
        `alpha`: The step size for the extragradient algorithm.
        `max_iter`: The maximum number of iterations.
        `tol`: The tolerance for the solution.
        `dtype`: The data type of the solution.

    Returns:
        `EGState`: The state of the extragradient algorithm.
    """
    vec_c = jnp.asarray(vec_c, dtype=dtype)
    mat_A = jnp.asarray(mat_A, dtype=dtype)
    vec_b = jnp.asarray(vec_b, dtype=dtype)
    init_state = init_state_generate(
        vec_c=vec_c,
        mat_A=mat_A,
        vec_b=vec_b,
        vec_x_init=vec_x_init,
        vec_y_init=vec_y_init,
        alpha=alpha,
        dtype=dtype,
        max_iter=max_iter,
        tol=tol,
    )
    final_state = lax.while_loop(
        _cond_fn,
        _update_fn,
        init_state,
    )
    return final_state


def init_state_generate(
    vec_c: jnp.ndarray,
    mat_A: jnp.ndarray,
    vec_b: jnp.ndarray,
    vec_x_init: Optional[jnp.ndarray] = None,
    vec_y_init: Optional[jnp.ndarray] = None,
    alpha: Optional[float] = None,
    dtype: jnp.dtype = jnp.float32,
    max_iter: int = 5000,
    tol: float = 1e-4,
) -> EGState:
    # Initialize primal and dual variables
    if vec_x_init is None:
        vec_x_init = jnp.zeros(mat_A.shape[1], dtype=dtype)
    if vec_y_init is None:
        vec_y_init = jnp.zeros(mat_A.shape[0], dtype=dtype)

    if alpha is None:
        alpha = 1.0 / (jnp.linalg.norm(mat_A, ord=2) + 1e-8)

    state = EGState(
        vec_c=vec_c,
        mat_A=mat_A,
        vec_b=vec_b,
        vec_x=vec_x_init,
        vec_y=vec_y_init,
        alpha=alpha,
        curr_iter=0,
        max_iter=max_iter,
        tol=tol,
    )
    return state


def _cond_fn(eg_state: EGState) -> bool:
    """Condition for the PDHG solver to continue iterating."""
    logit1 = eg_state.curr_iter < eg_state.max_iter
    logit2 = jnp.linalg.norm(
        jnp.dot(eg_state.mat_A, eg_state.vec_x) - eg_state.vec_b
    ) > eg_state.tol * (1.0 + jnp.linalg.norm(eg_state.vec_b))
    lambd = eg_state.vec_c - jnp.dot(eg_state.mat_A.T, eg_state.vec_y)
    lambd = jnp.clip(lambd, None, 0)
    logit3 = jnp.linalg.norm(lambd) > eg_state.tol * (
        1.0 + jnp.linalg.norm(eg_state.vec_c)
    )
    logit4 = jnp.abs(
        jnp.vdot(eg_state.vec_b, eg_state.vec_y)
        - jnp.vdot(eg_state.vec_x, eg_state.vec_c)
    ) > eg_state.tol * (
        1.0
        + jnp.abs(jnp.vdot(eg_state.vec_b, eg_state.vec_y))
        + jnp.abs(jnp.vdot(eg_state.vec_x, eg_state.vec_c))
    )
    return jnp.logical_and(logit1, jnp.any(jnp.asarray([logit2, logit3, logit4])))


def _update_fn(eg_state: EGState) -> EGState:
    """Update the state of the extragradient algorithm."""
    alpha = eg_state.alpha
    vec_x_new = eg_state.vec_x - alpha * (
        eg_state.vec_c - jnp.dot(eg_state.mat_A.T, eg_state.vec_y)
    )
    vec_x_new = jnp.clip(vec_x_new, 0, None)
    vec_y_new = eg_state.vec_y + alpha * (
        eg_state.vec_b - jnp.dot(eg_state.mat_A, eg_state.vec_x)
    )
    vec_x_new = eg_state.vec_x - alpha * (
        eg_state.vec_c - jnp.dot(eg_state.mat_A.T, vec_y_new)
    )
    vec_x_new = jnp.clip(vec_x_new, 0, None)
    vec_y_new = eg_state.vec_y + alpha * (
        eg_state.vec_b - jnp.dot(eg_state.mat_A, vec_x_new)
    )

    new_state = EGState(
        vec_c=eg_state.vec_c,
        mat_A=eg_state.mat_A,
        vec_b=eg_state.vec_b,
        vec_x=vec_x_new,
        vec_y=vec_y_new,
        alpha=alpha,
        curr_iter=eg_state.curr_iter + 1,
        max_iter=eg_state.max_iter,
        tol=eg_state.tol,
    )
    return new_state
