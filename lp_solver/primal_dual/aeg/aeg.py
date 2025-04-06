from typing import Optional

import chex
import jax.numpy as jnp
from jax import lax


@chex.dataclass
class AEGVState:
    vec_c: jnp.ndarray
    mat_A: jnp.ndarray
    vec_b: jnp.ndarray
    vec_x_x: jnp.ndarray
    vec_x_y: jnp.ndarray
    vec_y_x: jnp.ndarray
    vec_y_y: jnp.ndarray
    vec_w_x: jnp.ndarray
    vec_w_y: jnp.ndarray
    eta: float
    curr_iter: int
    max_iter: int
    tol: float


def aeg_solve(
    vec_c: jnp.ndarray,
    mat_A: jnp.ndarray,
    vec_b: jnp.ndarray,
    vec_x_init: Optional[jnp.ndarray] = None,
    vec_y_init: Optional[jnp.ndarray] = None,
    eta: Optional[float] = None,
    max_iter: int = 5000,
    tol: float = 1e-4,
    dtype: jnp.dtype = jnp.float32,
) -> AEGVState:
    """
    Solve the linear programming problem using the accelerated extragradient algorithm.
    See the paper: https://arxiv.org/pdf/2302.04099.pdf

    Args:
        `vec_c`: jnp.ndarray, shape (n,)
        `mat_A`: jnp.ndarray, shape (m, n)
        `vec_b`: jnp.ndarray, shape (m,)
        `vec_x_init`: jnp.ndarray, shape (n,)
        `vec_y_init`: jnp.ndarray, shape (m,)
        `eta`: float
        `max_iter`: int
        `tol`: float
        `dtype`: jnp.dtype

    Returns:
        `AEGVState`: AEGVState
    """
    state = init_state_generate(
        vec_c=vec_c,
        mat_A=mat_A,
        vec_b=vec_b,
        vec_x_init=vec_x_init,
        vec_y_init=vec_y_init,
        eta=eta,
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
    eta: Optional[float] = None,
    dtype: jnp.dtype = jnp.float32,
    max_iter: int = 5000,
    tol: float = 1e-4,
) -> AEGVState:
    # Initialize primal and dual variables
    if vec_x_init is None:
        vec_x_init = jnp.zeros(mat_A.shape[1], dtype=dtype)
    if vec_y_init is None:
        vec_y_init = jnp.zeros(mat_A.shape[0], dtype=dtype)
    if eta is None:
        eta = 1.0 / (jnp.linalg.norm(mat_A, ord=2) + 1e-8)

    vec_w_x = vec_c - jnp.dot(mat_A.T, vec_y_init)
    vec_w_y = -vec_b + jnp.dot(mat_A, vec_x_init)

    state = AEGVState(
        vec_c=vec_c,
        mat_A=mat_A,
        vec_b=vec_b,
        vec_x_x=vec_x_init,
        vec_x_y=vec_y_init,
        vec_y_x=vec_x_init,
        vec_y_y=vec_y_init,
        vec_w_x=vec_w_x,
        vec_w_y=vec_w_y,
        eta=eta,
        curr_iter=0,
        max_iter=max_iter,
        tol=tol,
    )
    return state


def _cond_fn(aeg_state: AEGVState) -> bool:
    """Condition for the PDHG solver to continue iterating."""
    vec_x = aeg_state.vec_x_x
    vec_y = aeg_state.vec_x_y

    logit1 = aeg_state.curr_iter < aeg_state.max_iter
    logit2 = jnp.linalg.norm(
        jnp.dot(aeg_state.mat_A, vec_x) - aeg_state.vec_b
    ) > aeg_state.tol * (1.0 + jnp.linalg.norm(aeg_state.vec_b))
    lambd = aeg_state.vec_c - jnp.dot(aeg_state.mat_A.T, vec_y)
    lambd = jnp.clip(lambd, None, 0)
    logit3 = jnp.linalg.norm(lambd) > aeg_state.tol * (
        1.0 + jnp.linalg.norm(aeg_state.vec_c)
    )
    logit4 = jnp.abs(
        jnp.vdot(aeg_state.vec_b, vec_y) - jnp.vdot(vec_x, aeg_state.vec_c)
    ) > aeg_state.tol * (
        1.0
        + jnp.abs(jnp.vdot(aeg_state.vec_b, vec_y))
        + jnp.abs(jnp.vdot(vec_x, aeg_state.vec_c))
    )
    return jnp.logical_and(logit1, jnp.any(jnp.asarray([logit2, logit3, logit4])))


def _update_fn(aeg_state: AEGVState) -> AEGVState:
    """Update the state of the accelerated extragradient algorithm."""
    eta = aeg_state.eta
    curr_iter = aeg_state.curr_iter
    mat_A = aeg_state.mat_A
    vec_b = aeg_state.vec_b
    vec_c = aeg_state.vec_c

    # update parameters
    t_k = curr_iter + 2
    eta_hat = eta * (t_k - 1) / t_k
    theta_k = (t_k - 1) / (t_k + 1)
    nu_k = t_k / (t_k + 1)
    beta_k = eta * (1.0 + theta_k - 2.0 * nu_k)
    sigma_k = eta * theta_k - nu_k * eta_hat


    # update the variable x.
    f_y_x = vec_c - jnp.dot(mat_A.T, aeg_state.vec_y_y)
    f_y_y = -vec_b + jnp.dot(mat_A, aeg_state.vec_y_x)
    vec_x_x_new = aeg_state.vec_y_x - eta * f_y_x 
    vec_x_x_new = jnp.clip(vec_x_x_new, 0, None)
    vec_x_y_new = aeg_state.vec_y_y - eta * f_y_y

    # update the variable y.
    f_x_x = vec_c - jnp.dot(mat_A.T, vec_x_y_new)
    f_x_y = -vec_b + jnp.dot(mat_A, vec_x_x_new)
    vec_y_x_new = vec_x_x_new - (eta * f_x_x - eta * f_y_x)
    vec_y_x_new = jnp.clip(vec_y_x_new, 0, None)
    vec_y_y_new = vec_x_y_new - (eta * f_x_y - eta * f_y_y)

    return AEGVState(
        vec_c=vec_c,
        mat_A=mat_A,
        vec_b=vec_b,
        vec_x_x=vec_x_x_new,
        vec_x_y=vec_x_y_new,
        vec_y_x=vec_y_x_new,
        vec_y_y=vec_y_y_new,
        vec_w_x=aeg_state.vec_w_x,
        vec_w_y=aeg_state.vec_w_y,
        eta=eta,
        curr_iter=curr_iter + 1,
        max_iter=aeg_state.max_iter,
        tol=aeg_state.tol,
    )

    # update the variable x.
    f_y_x = vec_c - jnp.dot(mat_A.T, aeg_state.vec_y_y)
    f_y_y = -vec_b + jnp.dot(mat_A, aeg_state.vec_y_x)
    vec_x_x_new = aeg_state.vec_y_x - eta * f_y_x + eta_hat * aeg_state.vec_w_x
    vec_x_x_new = jnp.clip(vec_x_x_new, 0, None)
    vec_x_y_new = aeg_state.vec_y_y - eta * f_y_y + eta_hat * aeg_state.vec_w_y

    # update the variable w.
    f_x_x = vec_c - jnp.dot(mat_A.T, vec_x_y_new)
    f_x_y = -vec_b + jnp.dot(mat_A, vec_x_x_new)
    tmp_x = f_x_x - f_y_x
    tmp_y = f_x_y - f_y_y
    vec_w_x_new = (
        aeg_state.vec_y_x - vec_x_x_new + eta_hat * aeg_state.vec_w_x
    ) / eta + tmp_x
    vec_w_y_new = (
        aeg_state.vec_y_y - vec_x_y_new + eta_hat * aeg_state.vec_w_y
    ) / eta + tmp_y

    # update the variable y.
    vec_y_x_new = (
        vec_x_x_new
        + theta_k * (vec_x_x_new - aeg_state.vec_x_x)
        - beta_k * vec_w_x_new
        + sigma_k * aeg_state.vec_w_x
        - tmp_x * eta * nu_k
    )
    vec_y_y_new = (
        vec_x_y_new
        + theta_k * (vec_x_y_new - aeg_state.vec_x_y)
        - beta_k * vec_w_y_new
        + sigma_k * tmp_y * aeg_state.vec_w_y
        - tmp_y * eta * nu_k
    )

    return AEGVState(
        vec_c=vec_c,
        mat_A=mat_A,
        vec_b=vec_b,
        vec_x_x=vec_x_x_new,
        vec_x_y=vec_x_y_new,
        vec_y_x=vec_y_x_new,
        vec_y_y=vec_y_y_new,
        vec_w_x=vec_w_x_new,
        vec_w_y=vec_w_y_new,
        eta=eta,
        curr_iter=curr_iter + 1,
        max_iter=aeg_state.max_iter,
        tol=aeg_state.tol,
    )
