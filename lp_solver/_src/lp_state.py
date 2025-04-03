from dataclasses import dataclass
from typing import NamedTuple

import jax 
import jax.numpy as jnp


@dataclass
class StandardLPState:
    r"""
    A named tuple representing the state of a linear programming problem.
    The stateclass of the linear programming problem is defined in the standard form:

    .. math::
        \begin{align*}
        \text{maximize} & \quad c^T x \\
        \text{subject to} & \quad Ax = b \\
                          & \quad x \geq 0
        \end{align*}

    Attributes:
        `vec_c`: Coefficients of the objective function.
        `mat_A`: Coefficient matrix of the constraints.
        `vec_b`: Right-hand side vector of the constraints.
        `vec_x`: Current solution vector.
    """
    vec_c: jnp.ndarray
    mat_A: jnp.ndarray
    vec_b: jnp.ndarray
    vec_x: jnp.ndarray