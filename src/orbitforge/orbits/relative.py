from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from orbitforge.core.constants import MU_EARTH_KM3_S2
from orbitforge.core.vector import Vec3
from orbitforge.orbits.elements import state_to_elements
from orbitforge.orbits.kepler import (
    eccentric_from_true,
    solve_kepler_elliptic,
    true_from_eccentric,
)
from orbitforge.orbits.universal import propagate_universal

# Below this eccentricity the analytic C/W-Hill transition matrix is used.
# The eccentric formulation has the same limit, but using the closed form here
# avoids a numerically divided integration step and makes e == 0 bit-stable.
_CIRCULAR_ECCENTRICITY = 1.0e-12


@dataclass(frozen=True)
class EccentricChief:
    """Keplerian reference orbit used by the Tschauner--Hempel model.

    Only in-plane scalar elements are needed for the relative state-transition
    matrix.  Orientation is supplied separately when rotating back to inertial
    Cartesian states.
    """

    a_km: float
    e: float
    mean_motion_rad_s: float
    eccentric_anomaly_rad: float

    def __post_init__(self):
        if self.a_km <= 0.0:
            raise ValueError("chief semimajor axis must be positive")
        if not 0.0 <= self.e < 1.0:
            raise ValueError("chief eccentricity must satisfy 0 <= e < 1")
        if self.mean_motion_rad_s <= 0.0:
            raise ValueError("chief mean motion must be positive")

    @classmethod
    def from_state(
        cls,
        r: Vec3,
        v: Vec3,
        mu: float = MU_EARTH_KM3_S2,
    ) -> "EccentricChief":
        elements = state_to_elements(r, v, mu)
        if elements.e >= 1.0:
            raise ValueError("eccentric relative model requires an elliptic chief")
        return cls(
            a_km=elements.a_km,
            e=elements.e,
            mean_motion_rad_s=math.sqrt(mu / elements.a_km**3),
            eccentric_anomaly_rad=eccentric_from_true(elements.nu_rad, elements.e),
        )

    @property
    def period_s(self) -> float:
        return 2.0 * math.pi / self.mean_motion_rad_s

    def radius_at(self, eccentric_anomaly_rad: float) -> float:
        e = self.e
        return self.a_km * (1.0 - e * math.cos(eccentric_anomaly_rad))

    def true_anomaly_at(self, eccentric_anomaly_rad: float) -> float:
        return true_from_eccentric(eccentric_anomaly_rad, self.e)

    def angular_rate_at(self, eccentric_anomaly_rad: float) -> float:
        q = 1.0 - self.e * math.cos(eccentric_anomaly_rad)
        return self.mean_motion_rad_s * math.sqrt(1.0 - self.e * self.e) / (q * q)

    def time_at_eccentric_anomaly(self, eccentric_anomaly_rad: float) -> float:
        """Elapsed time from this object's epoch to an unwrapped anomaly."""
        e = self.e
        m = eccentric_anomaly_rad - e * math.sin(eccentric_anomaly_rad)
        m0 = self.eccentric_anomaly_rad - e * math.sin(self.eccentric_anomaly_rad)
        return (m - m0) / self.mean_motion_rad_s

    def eccentric_anomaly_after(self, dt_s: float) -> float:
        return eccentric_anomaly_after(self, dt_s)


@dataclass(frozen=True)
class EccentricRelativePoint:
    time_s: float
    eccentric_anomaly_rad: float
    position_km: Vec3
    # Components of d(position)/dt expressed in the instantaneous R/T/N frame.
    rotating_velocity_km_s: Vec3
    # Components of the inertial relative velocity vector in R/T/N components.
    inertial_velocity_km_s: Vec3
    # Maps [r_RTN, d(r_RTN)/dt] at epoch to the same convention at time_s.
    state_transition_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class RelativePropagationError:
    position_error_km: float
    velocity_error_km_s: float
    relative_position_error: float
    linear_position_km: Vec3
    nonlinear_position_km: Vec3
    linear_velocity_km_s: Vec3
    nonlinear_velocity_km_s: Vec3


@dataclass(frozen=True)
class CircularProxyError:
    """Error made by forcing a circular C/W-Hill model onto an eccentric chief."""

    eccentricity: float
    mean_motion_used_rad_s: float
    position_error_km: float
    velocity_error_km_s: float
    nonlinear_position_km: Vec3
    circular_proxy_position_km: Vec3


def hill_frame(chief_r: Vec3, chief_v: Vec3):
    x = chief_r.unit()
    z = chief_r.cross(chief_v).unit()
    y = z.cross(x)
    return (x, y, z)


def relative_hill(chief_r, chief_v, deputy_r, deputy_v):
    """Return Hill position and inertial-velocity component differences.

    The second vector is the inertial relative velocity projected onto R/T/N.
    It is not d(relative Hill position)/dt; use
    :func:`rotating_relative_velocity` to make that frame-rate conversion.
    """
    x, y, z = hill_frame(chief_r, chief_v)
    dr = deputy_r - chief_r
    dv = deputy_v - chief_v
    return (Vec3(dr.dot(x), dr.dot(y), dr.dot(z)), Vec3(dv.dot(x), dv.dot(y), dv.dot(z)))


def rotating_relative_velocity(
    chief: EccentricChief,
    rel_r: Vec3,
    inertial_velocity_components: Vec3,
) -> Vec3:
    """Convert inertial R/T/N velocity components to rotating-frame derivative."""
    omega = chief.angular_rate_at(chief.eccentric_anomaly_rad)
    return Vec3(
        inertial_velocity_components.x + omega * rel_r.y,
        inertial_velocity_components.y - omega * rel_r.x,
        inertial_velocity_components.z,
    )


def inertial_relative_velocity(
    chief: EccentricChief,
    eccentric_anomaly_rad: float,
    rel_r: Vec3,
    rotating_velocity: Vec3,
) -> Vec3:
    """Convert d(Hill position)/dt to inertial velocity R/T/N components."""
    omega = chief.angular_rate_at(eccentric_anomaly_rad)
    return Vec3(
        rotating_velocity.x - omega * rel_r.y,
        rotating_velocity.y + omega * rel_r.x,
        rotating_velocity.z,
    )


def cwh_propagate(rel_r: Vec3, rel_v: Vec3, n: float, t: float):
    """Closed-form circular chief C/W-Hill position propagation.

    ``rel_v`` is the rotating-frame derivative.  Use
    :func:`rotating_relative_velocity` to convert an inertial R/T/N velocity
    difference into this C/W state convention.  Callers with eccentric chief
    states should use :func:`eccentric_relative_propagate` instead.
    """
    c, s = (math.cos(n * t), math.sin(n * t))
    x = (4 - 3 * c) * rel_r.x + s / n * rel_v.x + 2 * (1 - c) / n * rel_v.y
    y = (
        6 * (s - n * t) * rel_r.x
        + rel_r.y
        - 2 * (1 - c) / n * rel_v.x
        + (4 * s - 3 * n * t) / n * rel_v.y
    )
    z = c * rel_r.z + s / n * rel_v.z
    return Vec3(x, y, z)


def _cwh_state_transition_matrix(n: float, t: float) -> tuple[tuple[float, ...], ...]:
    nt = n * t
    c = math.cos(nt)
    s = math.sin(nt)
    return (
        (4.0 - 3.0 * c, 0.0, 0.0, s / n, 2.0 * (1.0 - c) / n, 0.0),
        (6.0 * (s - nt), 1.0, 0.0, -2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * nt) / n, 0.0),
        (0.0, 0.0, c, 0.0, 0.0, s / n),
        (3.0 * n * s, 0.0, 0.0, c, 2.0 * s, 0.0),
        (6.0 * n * (c - 1.0), 0.0, 0.0, -2.0 * s, 4.0 * c - 3.0, 0.0),
        (0.0, 0.0, -n * s, 0.0, 0.0, c),
    )


def eccentric_anomaly_after(chief: EccentricChief, dt_s: float) -> float:
    """Solve Kepler's equation while retaining an unwrapped anomaly branch."""
    e = chief.e
    n = chief.mean_motion_rad_s
    target_mean = (
        chief.eccentric_anomaly_rad
        - e * math.sin(chief.eccentric_anomaly_rad)
        + n * dt_s
    )
    if e < 1.0e-14:
        return chief.eccentric_anomaly_rad + n * dt_s

    wrapped = solve_kepler_elliptic(target_mean, e)
    guess = chief.eccentric_anomaly_rad + n * dt_s / max(
        1.0e-14, 1.0 - e * math.cos(chief.eccentric_anomaly_rad)
    )
    anomaly = wrapped + 2.0 * math.pi * round((guess - wrapped) / (2.0 * math.pi))

    # Newton refinement on the unwrapped branch.
    for _ in range(20):
        f = anomaly - e * math.sin(anomaly) - target_mean
        fp = 1.0 - e * math.cos(anomaly)
        step = f / fp
        anomaly -= step
        if abs(step) < 1.0e-13:
            break
    return anomaly


def _eccentric_matrix_derivative(
    chief: EccentricChief,
    eccentric_anomaly_rad: float,
) -> list[list[float]]:
    """d(state)/dE for physical R/T/N coordinates.

    The state is [x, y, z, xdot, ydot, zdot], where dot is time derivative in
    the rotating Hill frame.  This form of Tschauner--Hempel reduces to C/W-Hill
    as e -> 0 rather than inserting an instantaneous rate into circular code.
    """
    e = chief.e
    n = chief.mean_motion_rad_s
    E = eccentric_anomaly_rad
    q = 1.0 - e * math.cos(E)
    s = math.sqrt(1.0 - e * e)

    return [
        [0.0, 0.0, 0.0, q / n, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, q / n, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, q / n],
        [
            n * (s * s + 2.0 * q) / q**3,
            -2.0 * n * s * e * math.sin(E) / q**3,
            0.0,
            0.0,
            2.0 * s / q,
            0.0,
        ],
        [
            2.0 * n * s * e * math.sin(E) / q**3,
            n * e * (math.cos(E) - e) / q**3,
            0.0,
            -2.0 * s / q,
            0.0,
            0.0,
        ],
        [0.0, 0.0, -n / (q * q), 0.0, 0.0, 0.0],
    ]


def _identity6() -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]


def _flatten6(matrix: list[list[float]]) -> list[float]:
    return [value for row in matrix for value in row]


def _unflatten6(values: list[float]) -> list[list[float]]:
    return [values[i * 6 : (i + 1) * 6] for i in range(6)]


def _matvec6(matrix: Iterable[Iterable[float]], vector: list[float]) -> list[float]:
    return [sum(row[j] * vector[j] for j in range(6)) for row in matrix]


def _matmul6(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(6)) for j in range(6)] for i in range(6)
    ]


def _rk4_eccentric_state_and_stm_once(
    chief: EccentricChief,
    state0: list[float],
    final_anomaly_rad: float,
    steps_per_orbit: int,
) -> tuple[list[float], tuple[tuple[float, ...], ...]]:
    e0 = chief.eccentric_anomaly_rad
    delta = final_anomaly_rad - e0
    orbits = abs(delta) / (2.0 * math.pi)
    # More uniform anomaly steps are useful near perigee for high eccentricity.
    spacing_factor = max(0.2, 1.0 - chief.e)
    steps = max(16, math.ceil(steps_per_orbit * orbits / spacing_factor))
    h = delta / steps

    def derivative(E: float, augmented: list[float]) -> list[float]:
        current_state = augmented[:6]
        phi = _unflatten6(augmented[6:])
        a = _eccentric_matrix_derivative(chief, E)
        state_dot = _matvec6(a, current_state)
        phi_dot = _matmul6(a, phi)
        return state_dot + _flatten6(phi_dot)

    augmented = state0 + _flatten6(_identity6())
    E = e0
    for _ in range(steps):
        k1 = derivative(E, augmented)
        k2 = derivative(E + h / 2.0, [v + h / 2.0 * k for v, k in zip(augmented, k1)])
        k3 = derivative(E + h / 2.0, [v + h / 2.0 * k for v, k in zip(augmented, k2)])
        k4 = derivative(E + h, [v + h * k for v, k in zip(augmented, k3)])
        augmented = [
            value + h / 6.0 * (a + 2.0 * b + 2.0 * c + d)
            for value, a, b, c, d in zip(augmented, k1, k2, k3, k4)
        ]
        E += h

    final_state = augmented[:6]
    stm = tuple(tuple(row) for row in _unflatten6(augmented[6:]))
    return final_state, stm


def _rk4_eccentric_state_and_stm(
    chief: EccentricChief,
    state0: list[float],
    final_anomaly_rad: float,
    steps_per_orbit: int,
) -> tuple[list[float], tuple[tuple[float, ...], ...]]:
    """Richardson-extrapolated RK4 integration of state and STM."""
    coarse_state, coarse_stm = _rk4_eccentric_state_and_stm_once(
        chief, state0, final_anomaly_rad, steps_per_orbit
    )
    fine_state, fine_stm = _rk4_eccentric_state_and_stm_once(
        chief, state0, final_anomaly_rad, 2 * steps_per_orbit
    )
    state = [
        fine + (fine - coarse) / 15.0
        for coarse, fine in zip(coarse_state, fine_state)
    ]
    stm = tuple(
        tuple(
            fine + (fine - coarse) / 15.0
            for coarse, fine in zip(coarse_row, fine_row)
        )
        for coarse_row, fine_row in zip(coarse_stm, fine_stm)
    )
    return state, stm


def _state_to_vectors(state: list[float]) -> tuple[Vec3, Vec3]:
    return (
        Vec3(state[0], state[1], state[2]),
        Vec3(state[3], state[4], state[5]),
    )


def eccentric_relative_propagate(
    chief: EccentricChief,
    rel_r0_km: Vec3,
    rel_v0_km_s: Vec3,
    dt_s: float,
    steps_per_orbit: int = 256,
) -> EccentricRelativePoint:
    """Propagate linearized relative motion about an eccentric Keplerian chief.

    This integrates Tschauner--Hempel equations in eccentric anomaly.  The
    input velocity is d(r_RTN)/dt in the rotating frame.  Use
    :func:`rotating_relative_velocity` when starting from inertial Cartesian
    chief/deputy states.
    """
    if steps_per_orbit < 16:
        raise ValueError("steps_per_orbit must be at least 16")

    final_anomaly = eccentric_anomaly_after(chief, dt_s)
    state0 = [
        rel_r0_km.x,
        rel_r0_km.y,
        rel_r0_km.z,
        rel_v0_km_s.x,
        rel_v0_km_s.y,
        rel_v0_km_s.z,
    ]

    if chief.e <= _CIRCULAR_ECCENTRICITY:
        phi = _cwh_state_transition_matrix(chief.mean_motion_rad_s, dt_s)
        final_state = _matvec6(phi, state0)
    else:
        final_state, phi = _rk4_eccentric_state_and_stm(
            chief, state0, final_anomaly, steps_per_orbit
        )

    final_r, final_v = _state_to_vectors(final_state)
    inertial_v = inertial_relative_velocity(chief, final_anomaly, final_r, final_v)
    return EccentricRelativePoint(
        time_s=dt_s,
        eccentric_anomaly_rad=final_anomaly,
        position_km=final_r,
        rotating_velocity_km_s=final_v,
        inertial_velocity_km_s=inertial_v,
        state_transition_matrix=phi,
    )


def eccentric_relative_propagate_from_states(
    chief_r0: Vec3,
    chief_v0: Vec3,
    deputy_r0: Vec3,
    deputy_v0: Vec3,
    dt_s: float,
    mu: float = MU_EARTH_KM3_S2,
    steps_per_orbit: int = 256,
) -> EccentricRelativePoint:
    """Propagate relative motion directly from inertial chief/deputy states."""
    chief = EccentricChief.from_state(chief_r0, chief_v0, mu)
    rel_r0, rel_v_inertial0 = relative_hill(
        chief_r0, chief_v0, deputy_r0, deputy_v0
    )
    rel_v0 = rotating_relative_velocity(chief, rel_r0, rel_v_inertial0)
    return eccentric_relative_propagate(
        chief, rel_r0, rel_v0, dt_s, steps_per_orbit=steps_per_orbit
    )


def eccentric_relative_stm(
    chief: EccentricChief,
    dt_s: float,
    steps_per_orbit: int = 256,
) -> tuple[tuple[float, ...], ...]:
    point = eccentric_relative_propagate(
        chief,
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 0.0),
        dt_s,
        steps_per_orbit,
    )
    return point.state_transition_matrix


def _basis_vectors_to_inertial(basis: tuple[Vec3, Vec3, Vec3], vector: Vec3) -> Vec3:
    return basis[0] * vector.x + basis[1] * vector.y + basis[2] * vector.z


def nonlinear_relative_error(
    chief_r0: Vec3,
    chief_v0: Vec3,
    deputy_r0: Vec3,
    deputy_v0: Vec3,
    dt_s: float,
    mu: float = MU_EARTH_KM3_S2,
    steps_per_orbit: int = 256,
) -> RelativePropagationError:
    """Compare eccentric linear propagation with exact two-body propagation.

    The reported error is the truncation error of the first-order relative
    model under point-mass gravity, generally O((rho/r)^2) in position plus
    any secular contribution if a very large separation makes the linearization
    region inappropriate.  It does not include J2, drag, third-body effects, or
    other external perturbations.
    """
    chief = EccentricChief.from_state(chief_r0, chief_v0, mu)
    rel_r0, rel_v_inertial0 = relative_hill(
        chief_r0, chief_v0, deputy_r0, deputy_v0
    )
    rel_v0 = rotating_relative_velocity(chief, rel_r0, rel_v_inertial0)

    linear = eccentric_relative_propagate(
        chief, rel_r0, rel_v0, dt_s, steps_per_orbit=steps_per_orbit
    )

    chief_r1, chief_v1 = propagate_universal(chief_r0, chief_v0, dt_s, mu)
    deputy_r1, deputy_v1 = propagate_universal(deputy_r0, deputy_v0, dt_s, mu)
    exact_r, exact_v_inertial = relative_hill(
        chief_r1, chief_v1, deputy_r1, deputy_v1
    )

    dr = linear.position_km - exact_r
    dv = linear.inertial_velocity_km_s - exact_v_inertial
    chief_radius = max(chief_r1.norm(), 1.0e-12)
    return RelativePropagationError(
        position_error_km=dr.norm(),
        velocity_error_km_s=dv.norm(),
        relative_position_error=dr.norm() / chief_radius,
        linear_position_km=linear.position_km,
        nonlinear_position_km=exact_r,
        linear_velocity_km_s=linear.inertial_velocity_km_s,
        nonlinear_velocity_km_s=exact_v_inertial,
    )


def instantaneous_circle_proxy_error(
    chief_r0: Vec3,
    chief_v0: Vec3,
    deputy_r0: Vec3,
    deputy_v0: Vec3,
    dt_s: float,
    mean_motion_rad_s: float | None = None,
    mu: float = MU_EARTH_KM3_S2,
) -> CircularProxyError:
    """Quantify the error from forcing C/W-Hill onto an eccentric chief.

    By default the local instantaneous rate ``sqrt(mu/r^3)`` is substituted
    into the circular formula.  The osculating mean motion ``sqrt(mu/a^3)`` can
    instead be passed explicitly.  Neither substitution reproduces T/H: it
    omits the periodic angular-rate and gravity-gradient terms driven by e.
    """
    chief = EccentricChief.from_state(chief_r0, chief_v0, mu)
    rel_r0, rel_v_inertial0 = relative_hill(
        chief_r0, chief_v0, deputy_r0, deputy_v0
    )
    rel_v0 = rotating_relative_velocity(chief, rel_r0, rel_v_inertial0)

    n_used = (
        math.sqrt(mu / chief_r0.norm() ** 3)
        if mean_motion_rad_s is None
        else mean_motion_rad_s
    )
    proxy_state = _matvec6(
        _cwh_state_transition_matrix(n_used, dt_s),
        [rel_r0.x, rel_r0.y, rel_r0.z, rel_v0.x, rel_v0.y, rel_v0.z],
    )
    proxy_r = Vec3(proxy_state[0], proxy_state[1], proxy_state[2])
    final_anomaly = eccentric_anomaly_after(chief, dt_s)
    proxy_v_rot = Vec3(proxy_state[3], proxy_state[4], proxy_state[5])
    proxy_v = inertial_relative_velocity(chief, final_anomaly, proxy_r, proxy_v_rot)

    chief_r1, chief_v1 = propagate_universal(chief_r0, chief_v0, dt_s, mu)
    deputy_r1, deputy_v1 = propagate_universal(deputy_r0, deputy_v0, dt_s, mu)
    exact_r, exact_v = relative_hill(chief_r1, chief_v1, deputy_r1, deputy_v1)

    return CircularProxyError(
        eccentricity=chief.e,
        mean_motion_used_rad_s=n_used,
        position_error_km=(proxy_r - exact_r).norm(),
        velocity_error_km_s=(proxy_v - exact_v).norm(),
        nonlinear_position_km=exact_r,
        circular_proxy_position_km=proxy_r,
    )
