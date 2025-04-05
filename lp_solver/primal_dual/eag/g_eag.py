from typing import Optional, Callable

import chex
import jax.numpy as jnp
from jax import lax

from lp_solver import StandardLPState


@chex.dataclass
class G_EAGState(StandardLPState):
    vec_y: jnp.ndarray
    vec_x_init: jnp.ndarray
    vec_y_init: jnp.ndarray
    alpha: float
    curr_iter: int
    max_iter: int
    tol: float


def g_eag_solve(
    vec_c: jnp.ndarray,
    mat_A: jnp.ndarray,
    vec_b: jnp.ndarray,
    vec_x_init: Optional[jnp.ndarray] = None,
    vec_y_init: Optional[jnp.ndarray] = None,
    alpha: Optional[float] = None,
    max_iter: int = 5000,
    tol: float = 1e-4,
    dtype: jnp.dtype = jnp.float32,
) -> G_EAGState:
    r"""Solve the standard linear programming problem using the extragradient accelerated gradient algorithm.

    The standard linear programming problem is of the form:
    min c^T x
    s.t. Ax = b
    x >= 0

    The gradient accelerated gradient algorithm is a primal-dual method for solving this problem.
    See like https://arxiv.org/pdf/2410.14369 for more details.

    Args:
        `vec_c`: The cost vector of the linear programming problem.
        `mat_A`: The constraint matrix of the linear programming problem.
        `vec_b`: The right-hand side vector of the linear programming problem.
        `vec_x_init`: The initial guess for the primal variable.
        `vec_y_init`: The initial guess for the dual variable.
        `alpha`: The step size for the gradient descent.
        `max_iter`: The maximum number of iterations.
        `tol`: The tolerance for the stopping criterion.
        `dtype`: The data type of the variables.

    Returns:
        `state`: The final state of the algorithm.
    """
    state = init_state_generate(
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
    final_state = lax.while_loop(_cond_fn, _update_fn, state)
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
) -> G_EAGState:
    # Initialize primal and dual variables
    if vec_x_init is None:
        vec_x_init = jnp.zeros(mat_A.shape[1], dtype=dtype)
    if vec_y_init is None:
        vec_y_init = jnp.zeros(mat_A.shape[0], dtype=dtype)

    if alpha is None:
        alpha = 1.0 / (jnp.linalg.norm(mat_A, ord=2) + 1e-8)

    state = G_EAGState(
        vec_c=vec_c,
        mat_A=mat_A,
        vec_b=vec_b,
        vec_x=vec_x_init,
        vec_y=vec_y_init,
        vec_x_init=vec_x_init,
        vec_y_init=vec_y_init,
        alpha=alpha,
        curr_iter=0,
        max_iter=max_iter,
        tol=tol,
    )
    return state


def _cond_fn(g_eag_state: G_EAGState) -> bool:
    """Condition for the PDHG solver to continue iterating."""
    logit1 = g_eag_state.curr_iter < g_eag_state.max_iter
    logit2 = jnp.linalg.norm(
        jnp.dot(g_eag_state.mat_A, g_eag_state.vec_x) - g_eag_state.vec_b
    ) > g_eag_state.tol * (1.0 + jnp.linalg.norm(g_eag_state.vec_b))
    lambd = g_eag_state.vec_c - jnp.dot(g_eag_state.mat_A.T, g_eag_state.vec_y)
    lambd = jnp.clip(lambd, None, 0)
    logit3 = jnp.linalg.norm(lambd) > g_eag_state.tol * (
        1.0 + jnp.linalg.norm(g_eag_state.vec_c)
    )
    logit4 = jnp.abs(
        jnp.vdot(g_eag_state.vec_b, g_eag_state.vec_y)
        - jnp.vdot(g_eag_state.vec_x, g_eag_state.vec_c)
    ) > g_eag_state.tol * (
        1.0
        + jnp.abs(jnp.vdot(g_eag_state.vec_b, g_eag_state.vec_y))
        + jnp.abs(jnp.vdot(g_eag_state.vec_x, g_eag_state.vec_c))
    )
    return jnp.logical_and(logit1, jnp.any(jnp.asarray([logit2, logit3, logit4])))


def _update_fn(g_eag_state: G_EAGState) -> G_EAGState:
    """Update the state of the gradient accelerated gradient algorithm."""
    alpha = g_eag_state.alpha
    curr_iter = g_eag_state.curr_iter
    epsilon = _epsilon_fn(curr_iter, alpha)

    vec_x_new = (1.0 - alpha * epsilon) * g_eag_state.vec_x - alpha * (
        g_eag_state.vec_c - jnp.dot(g_eag_state.mat_A.T, g_eag_state.vec_y)
    )
    vec_x_new = jnp.clip(vec_x_new, 0, None)
    vec_y_new = (1.0 - alpha * epsilon) * g_eag_state.vec_y + alpha * (
        g_eag_state.vec_b - jnp.dot(g_eag_state.mat_A, vec_x_new)
    )

    vec_x_new = (
        g_eag_state.vec_x
        - alpha
        * (g_eag_state.vec_c - jnp.dot(g_eag_state.mat_A.T, vec_y_new))
    ) / (1.0 + alpha * epsilon)
    vec_x_new = jnp.clip(vec_x_new, 0, None)
    vec_y_new = (
        g_eag_state.vec_y
        + alpha
        * (g_eag_state.vec_b - jnp.dot(g_eag_state.mat_A, vec_x_new))
    ) / (1.0 + alpha * epsilon)

    return G_EAGState(
        vec_c=g_eag_state.vec_c,
        mat_A=g_eag_state.mat_A,
        vec_b=g_eag_state.vec_b,
        vec_x=vec_x_new,
        vec_y=vec_y_new,
        vec_x_init=g_eag_state.vec_x_init,
        vec_y_init=g_eag_state.vec_y_init,
        alpha=alpha,
        curr_iter=curr_iter + 1,
        max_iter=g_eag_state.max_iter,
        tol=g_eag_state.tol,
    )


def _epsilon_fn(iter: int, alpha: float) -> float:
    return alpha / (iter + 1)
