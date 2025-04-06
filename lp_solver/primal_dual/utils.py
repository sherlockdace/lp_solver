from typing import Tuple

import jax.numpy as jnp


def _standard_monotone_fn(
    vec_x: jnp.ndarray,
    vec_y: jnp.ndarray,
    vec_c: jnp.ndarray,
    mat_A: jnp.ndarray,
    vec_b: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Standard monotone function for primal-dual linear programming.
    The Lagrangian function is given by:
    L(vec_x, vec_y) = vec_c^T vec_x - vec_y^T mat_A vec_x + vec_b^T vec_y.
    Then the monotone function is given by:
    f(vec_x, vec_y) = [vec_c - mat_A^T vec_y, -vec_b + mat_A vec_x]

    Args:
        `vec_x`: jnp.ndarray, shape (n,)
        `vec_y`: jnp.ndarray, shape (m,)
        `vec_c`: jnp.ndarray, shape (n,)
        `mat_A`: jnp.ndarray, shape (m, n)
        `vec_b`: jnp.ndarray, shape (m,)

    Returns:
        `vec_f_x`: jnp.ndarray, shape (n,)
        `vec_f_y`: jnp.ndarray, shape (m,)
    """
    return vec_c - jnp.dot(mat_A.T, vec_y), -vec_b + jnp.dot(mat_A, vec_x)
