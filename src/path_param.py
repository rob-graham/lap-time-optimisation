"""Path parameterisation utilities.

This module provides a cubic-spline based representation of a lateral
offset :math:`e(s)` from a reference centreline. It also exposes a helper
function to compute the curvature of the resulting path when the
centreline curvature is known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class LateralOffsetSpline:
    """Represent a lateral offset ``e(s)`` using a cubic spline.

    Parameters
    ----------
    s_control:
        Arc-length positions of the control points. Values must be
        strictly increasing.
    e_control:
        Lateral offset values at the control points.
    bc_type:
        Boundary condition passed to :class:`scipy.interpolate.CubicSpline`.
        Defaults to ``'natural'`` which enforces zero second derivatives at
        the boundaries.
    """

    s_control: Iterable[float]
    e_control: Iterable[float]
    bc_type: str | tuple = "natural"

    def __post_init__(self) -> None:
        s = np.asarray(self.s_control, dtype=float)
        e = np.asarray(self.e_control, dtype=float)
        if s.ndim != 1 or e.ndim != 1:
            raise ValueError("s_control and e_control must be one-dimensional")
        if s.size != e.size:
            raise ValueError("s_control and e_control must have the same length")
        if np.any(np.diff(s) <= 0):
            raise ValueError("s_control must be strictly increasing")
        self._spline = CubicSpline(s, e, bc_type=self.bc_type)

    def __call__(self, s: Iterable[float]) -> np.ndarray:
        """Evaluate the lateral offset at ``s``."""
        return self._spline(s)

    def first_derivative(self, s: Iterable[float]) -> np.ndarray:
        """Evaluate :math:`\frac{de}{ds}` at ``s``."""
        return self._spline(s, 1)

    def second_derivative(self, s: Iterable[float]) -> np.ndarray:
        """Evaluate :math:`\frac{d^2e}{ds^2}` at ``s``."""
        return self._spline(s, 2)


def path_curvature(
    s: Iterable[float],
    offset: LateralOffsetSpline,
    centreline_curvature: Iterable[float],
) -> np.ndarray:
    """Compute the curvature of a path defined by a lateral offset.

    The path is described by a reference centreline with curvature
    ``centreline_curvature`` and a lateral displacement ``e(s)`` defined by
    ``offset``. The resulting path curvature :math:`\\kappa` is computed from
    the Frenet-frame relations [1]_.

    Parameters
    ----------
    s:
        Arc-length positions at which to evaluate the curvature.
    offset:
        Spline representing the lateral offset ``e(s)``.
    centreline_curvature:
        Curvature of the reference centreline at ``s``.

    Returns
    -------
    numpy.ndarray
        Curvature of the resulting path at each ``s``.

    References
    ----------
    .. [1] Jokic, K. et al., "Racing Line Optimisation via Path
       Parameterisation", 2020.
    """
    s = np.asarray(s, dtype=float)
    kappa_c = np.asarray(centreline_curvature, dtype=float)
    if s.shape != kappa_c.shape:
        raise ValueError("s and centreline_curvature must have the same shape")

    e = offset(s)
    e_s = offset.first_derivative(s)
    e_ss = offset.second_derivative(s)
    kappa_c_s = np.gradient(kappa_c, s, edge_order=2)
    a = 1.0 - e * kappa_c

    numerator = kappa_c * a**2 + 2.0 * kappa_c * e_s**2 + e_ss * a + e * e_s * kappa_c_s
    denominator = (a**2 + e_s**2) ** 1.5
    return numerator / denominator
