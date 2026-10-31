"""Bollen dip types: phase-voltage phasor sets, symmetrical components and closed forms.

References: M. H. J. Bollen, Understanding Power Quality Problems (2000); Zhang & Bollen (2003).
All phasors are complex pu relative to the pre-event positive-sequence phasor (1 + j0) with
phase a as the special phase unless rotated. V is the characteristic voltage; a phase-angle
jump psi is applied by replacing V with V* = V * exp(j*psi).
"""

from __future__ import annotations

import numpy as np

from ..units import A_OP

A = A_OP
A2 = A_OP * A_OP
SQ3 = np.sqrt(3.0)

BOLLEN_TYPES = ("A", "B", "C", "D", "E", "F", "G")


def vstar(V: float, psi: float = 0.0) -> complex:
    """Complex characteristic voltage V* = V exp(j psi) (psi in rad)."""
    return complex(V * np.exp(1j * psi))


def bollen_phase_voltages(dip_type: str, V: complex) -> np.ndarray:
    """Phase-voltage phasors [Ua, Ub, Uc] of a Bollen dip type with phase a special (Bollen 2000).

    V may be complex (V* = V e^{j psi}). Expressions:
      A: Ua = V,          Ub = -V/2 - j(sqrt3/2)V,          Uc = -V/2 + j(sqrt3/2)V
      B: Ua = V,          Ub = -1/2 - j(sqrt3/2),           Uc = -1/2 + j(sqrt3/2)
      C: Ua = 1,          Ub = -1/2 - j(sqrt3/2)V,          Uc = -1/2 + j(sqrt3/2)V
      D: Ua = V,          Ub = -V/2 - j(sqrt3/2),           Uc = -V/2 + j(sqrt3/2)
      E: Ua = 1,          Ub = -V/2 - j(sqrt3/2)V,          Uc = -V/2 + j(sqrt3/2)V
      F: Ua = V,          Ub = -V/2 - j(2+V)/(2 sqrt3),     Uc = -V/2 + j(2+V)/(2 sqrt3)
      G: Ua = (2+V)/3,    Ub = -(2+V)/6 - j(sqrt3/2)V,      Uc = -(2+V)/6 + j(sqrt3/2)V
    """
    j = 1j
    h = SQ3 / 2.0
    if dip_type == "A":
        u = [V, -V / 2 - j * h * V, -V / 2 + j * h * V]
    elif dip_type == "B":
        u = [V, -0.5 - j * h, -0.5 + j * h]
    elif dip_type == "C":
        u = [1.0, -0.5 - j * h * V, -0.5 + j * h * V]
    elif dip_type == "D":
        u = [V, -V / 2 - j * h, -V / 2 + j * h]
    elif dip_type == "E":
        u = [1.0, -V / 2 - j * h * V, -V / 2 + j * h * V]
    elif dip_type == "F":
        u = [V, -V / 2 - j * (2 + V) / (2 * SQ3), -V / 2 + j * (2 + V) / (2 * SQ3)]
    elif dip_type == "G":
        u = [(2 + V) / 3, -(2 + V) / 6 - j * h * V, -(2 + V) / 6 + j * h * V]
    else:
        raise ValueError(f"unknown Bollen type {dip_type!r}")
    return np.asarray(u, dtype=complex)


def sequence_components(u: np.ndarray) -> tuple[complex, complex, complex]:
    """Fortescue: V0 = (Ua+Ub+Uc)/3, V1 = (Ua + a Ub + a^2 Uc)/3, V2 = (Ua + a^2 Ub + a Uc)/3."""
    ua, ub, uc = u
    v0 = (ua + ub + uc) / 3
    v1 = (ua + A * ub + A2 * uc) / 3
    v2 = (ua + A2 * ub + A * uc) / 3
    return complex(v0), complex(v1), complex(v2)


def phase_voltages_from_sequence(v0: complex, v1: complex, v2: complex) -> np.ndarray:
    """Inverse Fortescue: Ua = V0+V1+V2, Ub = V0 + a^2 V1 + a V2, Uc = V0 + a V1 + a^2 V2."""
    return np.asarray([v0 + v1 + v2, v0 + A2 * v1 + A * v2, v0 + A * v1 + A2 * v2], dtype=complex)


def rotate_special_phase(u: np.ndarray, special: str) -> np.ndarray:
    """Move the special phase from a to b or c in the phase domain.

    Special phase b: phase b takes the role of a (rotated by a^2), c the role of b, a the role of c:
      U'b = a^2 Ua, U'c = a^2 Ub, U'a = a^2 Uc.  Special phase c applies this twice.
    In sequence terms V1 is unchanged and V2 is multiplied by a (b) or a^2 (c).
    """
    u = np.asarray(u, dtype=complex)
    n = {"a": 0, "b": 1, "c": 2}[special]
    for _ in range(n):
        u = np.asarray([A2 * u[2], A2 * u[0], A2 * u[1]], dtype=complex)
    return u


def bollen_closed_form(dip_type: str, V: complex) -> tuple[complex, complex]:
    """Closed-form (V1, V2) of the zero-sequence-free types (brief B6.3), phase a special.

      A: V1 = V,          V2 = 0
      C: V1 = (1+V)/2,    V2 = (1-V)/2
      D: V1 = (1+V)/2,    V2 = -(1-V)/2
      F: V1 = (1+2V)/3,   V2 = -(1-V)/3
      G: V1 = (1+2V)/3,   V2 = (1-V)/3
    """
    if dip_type == "A":
        return complex(V), 0j
    if dip_type == "C":
        return complex((1 + V) / 2), complex((1 - V) / 2)
    if dip_type == "D":
        return complex((1 + V) / 2), complex(-(1 - V) / 2)
    if dip_type == "F":
        return complex((1 + 2 * V) / 3), complex(-(1 - V) / 3)
    if dip_type == "G":
        return complex((1 + 2 * V) / 3), complex((1 - V) / 3)
    raise ValueError(f"no closed form for type {dip_type!r} (B and E carry zero sequence)")


def dip_sequence(dip_type: str, V: float, psi: float, special: str) -> tuple[complex, complex]:
    """Relative (V1, V2) for a dip: closed form with V* = V e^{j psi}; V2 *= a (b) or a^2 (c)."""
    v1, v2 = bollen_closed_form(dip_type, vstar(V, psi))
    v2 *= {"a": 1.0, "b": A, "c": A2}[special]
    return v1, v2
