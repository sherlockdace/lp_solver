from dataclasses import dataclass
from typing import NamedTuple, Optional

import chex
import jax
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


@dataclass
class PDHGSolver:
    """PDHG solver for linear programming problems"""

    def __init__(
        self,
        max_iter: int = 5000,
        tol: float = 1e-4,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.dtype = dtype
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self,
        vec_c: jnp.ndarray,
        mat_A: jnp.ndarray,
        vec_b: jnp.ndarray,
        vec_x_init: Optional[jnp.ndarray] = None,
        vec_y_init: Optional[jnp.ndarray] = None,
        eta: Optional[float] = None,
        omega: Optional[float] = None,
    ) -> PDHGState:
        """Solve the linear programming problem using the PDHG algorithm."""
        state = self.init_state(vec_c, mat_A, vec_b, vec_x_init, vec_y_init, eta, omega)
        state = lax.while_loop(_cond_fn, _update_fn, state)
        return state

    def init_state(
        self,
        vec_c: jnp.ndarray,
        mat_A: jnp.ndarray,
        vec_b: jnp.ndarray,
        vec_x_init: Optional[jnp.ndarray] = None,
        vec_y_init: Optional[jnp.ndarray] = None,
        eta: Optional[float] = None,
        omega: Optional[float] = None,
    ) -> PDHGState:
        # Initialize primal and dual variables
        if vec_x_init is None:
            vec_x_init = jnp.zeros(mat_A.shape[1], dtype=self.dtype)
        if vec_y_init is None:
            vec_y_init = jnp.zeros(mat_A.shape[0], dtype=self.dtype)

        if eta is None:
            eta = 0.9 / jnp.linalg.norm(mat_A, ord=2)
        if omega is None:
            omega = 1.0

        # Initialize state
        state = PDHGState(
            vec_c=jnp.asarray(vec_c, dtype=self.dtype),
            mat_A=jnp.asarray(mat_A, dtype=self.dtype),
            vec_b=jnp.asarray(vec_b, dtype=self.dtype),
            vec_x=vec_x_init,
            vec_y=vec_y_init,
            eta=eta,
            omega=omega,
            curr_iter=0,
            max_iter=self.max_iter,
            tol=self.tol,
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
