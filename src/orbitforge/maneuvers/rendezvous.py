from __future__ import annotations

from dataclasses import dataclass, replace
import math

from orbitforge.core.constants import MU_EARTH_KM3_S2
from orbitforge.core.vector import Vec3
from orbitforge.orbits.relative import (
    EccentricChief,
    eccentric_relative_propagate,
    hill_frame,
    inertial_relative_velocity,
    relative_hill,
    rotating_relative_velocity,
)
from orbitforge.orbits.universal import propagate_universal


@dataclass(frozen=True)
class EccentricRendezvousSolution:
    time_of_flight_s: float
    final_eccentric_anomaly_rad: float
    initial_velocity_km_s: Vec3
    arrival_velocity_km_s: Vec3
    arrival_inertial_velocity_km_s: Vec3
    initial_delta_v_km_s: Vec3 | None
    state_transition_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class RendezvousOpportunity:
    start_time_s: float
    end_time_s: float
    minimum_time_s: float
    minimum_range_km: float


@dataclass(frozen=True)
class RendezvousValidation:
    position_miss_km: float
    velocity_miss_km_s: float
    final_relative_position_km: Vec3
    final_relative_velocity_km_s: Vec3


def _solve3(matrix: tuple[tuple[float, ...], ...], rhs: Vec3) -> Vec3:
    a = [[matrix[i][j] for j in range(3)] for i in range(3)]
    b = [rhs.x, rhs.y, rhs.z]
    scale = max(max(abs(v) for v in row) for row in a)
    if scale == 0.0:
        raise ValueError("singular rendezvous transfer")

    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) <= 1.0e-13 * scale:
            raise ValueError("singular rendezvous transfer")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            b[col], b[pivot] = b[pivot], b[col]
        for row in range(col + 1, 3):
            factor = a[row][col] / a[col][col]
            for k in range(col, 3):
                a[row][k] -= factor * a[col][k]
            b[row] -= factor * b[col]

    x = [0.0, 0.0, 0.0]
    for row in range(2, -1, -1):
        residual = sum(a[row][j] * x[j] for j in range(row + 1, 3))
        x[row] = (b[row] - residual) / a[row][row]
    return Vec3(x[0], x[1], x[2])


def _matvec3(matrix: tuple[tuple[float, ...], ...], vector: Vec3) -> Vec3:
    v = (vector.x, vector.y, vector.z)
    return Vec3(*(sum(row[j] * v[j] for j in range(3)) for row in matrix))


def _submatrix(
    matrix: tuple[tuple[float, ...], ...], rows: range, cols: range
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(matrix[i][j] for j in cols) for i in rows)


def cwh_targeting(initial_r: Vec3, target_r: Vec3, mean_motion_rad_s: float, tof_s: float):
    n = mean_motion_rad_s
    nt = n * tof_s
    c = math.cos(nt)
    s = math.sin(nt)
    a11 = s / n
    a12 = 2.0 * (1.0 - c) / n
    a21 = -2.0 * (1.0 - c) / n
    a22 = (4.0 * s - 3.0 * nt) / n
    bx = target_r.x - (4.0 - 3.0 * c) * initial_r.x
    by = target_r.y - (6.0 * (s - nt) * initial_r.x + initial_r.y)
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-12:
        raise ValueError('singular rendezvous transfer')
    vx = (bx * a22 - a12 * by) / det
    vy = (a11 * by - bx * a21) / det
    if abs(s) <= 1e-12:
        # At an out-of-plane half-period the boundary value is independent of
        # vz.  It is only reachable if the propagated z already matches.
        if abs(target_r.z - c * initial_r.z) > 1.0e-9:
            raise ValueError('singular out-of-plane rendezvous transfer')
        vz = 0.0
    else:
        vz = n * (target_r.z - c * initial_r.z) / s
    return Vec3(vx, vy, vz)


def eccentric_targeting(
    chief: EccentricChief,
    initial_r: Vec3,
    target_r: Vec3,
    tof_s: float,
    current_v: Vec3 | None = None,
    steps_per_orbit: int = 256,
) -> EccentricRendezvousSolution:
    """Solve a two-point relative transfer about an eccentric chief.

    Positions and velocities use the T/H state convention: R/T/N position and
    its rotating-frame time derivative.  The returned initial velocity is the
    required post-burn value; ``initial_delta_v_km_s`` is populated when the
    current relative velocity is supplied.
    """
    if tof_s <= 0.0:
        raise ValueError("rendezvous time of flight must be positive")

    zero = Vec3(0.0, 0.0, 0.0)
    propagation = eccentric_relative_propagate(
        chief, zero, zero, tof_s, steps_per_orbit=steps_per_orbit
    )
    phi = propagation.state_transition_matrix
    phi_rr = _submatrix(phi, range(0, 3), range(0, 3))
    phi_rv = _submatrix(phi, range(0, 3), range(3, 6))
    phi_vr = _submatrix(phi, range(3, 6), range(0, 3))
    phi_vv = _submatrix(phi, range(3, 6), range(3, 6))

    rhs = target_r - _matvec3(phi_rr, initial_r)
    required_v0 = _solve3(phi_rv, rhs)
    arrival_v = _matvec3(phi_vr, initial_r) + _matvec3(phi_vv, required_v0)
    arrival_inertial = inertial_relative_velocity(
        chief,
        propagation.eccentric_anomaly_rad,
        target_r,
        arrival_v,
    )
    delta_v = required_v0 - current_v if current_v is not None else None

    return EccentricRendezvousSolution(
        time_of_flight_s=tof_s,
        final_eccentric_anomaly_rad=propagation.eccentric_anomaly_rad,
        initial_velocity_km_s=required_v0,
        arrival_velocity_km_s=arrival_v,
        arrival_inertial_velocity_km_s=arrival_inertial,
        initial_delta_v_km_s=delta_v,
        state_transition_matrix=phi,
    )


def _range_at(
    chief: EccentricChief,
    rel_r: Vec3,
    rel_v: Vec3,
    time_s: float,
    steps_per_orbit: int,
) -> float:
    return eccentric_relative_propagate(
        chief, rel_r, rel_v, time_s, steps_per_orbit=steps_per_orbit
    ).position_km.norm()


def _bisect_crossing(
    chief: EccentricChief,
    rel_r: Vec3,
    rel_v: Vec3,
    outside_s: float,
    inside_s: float,
    threshold_km: float,
    steps_per_orbit: int,
) -> float:
    lo = outside_s
    hi = inside_s
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _range_at(chief, rel_r, rel_v, mid, steps_per_orbit) <= threshold_km:
            hi = mid
        else:
            lo = mid
    return hi


def _golden_minimum(
    chief: EccentricChief,
    rel_r: Vec3,
    rel_v: Vec3,
    lo_s: float,
    hi_s: float,
    steps_per_orbit: int,
) -> tuple[float, float]:
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi_s - inv_phi * (hi_s - lo_s)
    d = lo_s + inv_phi * (hi_s - lo_s)
    fc = _range_at(chief, rel_r, rel_v, c, steps_per_orbit)
    fd = _range_at(chief, rel_r, rel_v, d, steps_per_orbit)
    for _ in range(50):
        if fc < fd:
            hi_s = d
            d = c
            fd = fc
            c = hi_s - inv_phi * (hi_s - lo_s)
            fc = _range_at(chief, rel_r, rel_v, c, steps_per_orbit)
        else:
            lo_s = c
            c = d
            fc = fd
            d = lo_s + inv_phi * (hi_s - lo_s)
            fd = _range_at(chief, rel_r, rel_v, d, steps_per_orbit)
    t = 0.5 * (lo_s + hi_s)
    return t, _range_at(chief, rel_r, rel_v, t, steps_per_orbit)


def predict_rendezvous_opportunities(
    chief: EccentricChief,
    rel_r: Vec3,
    rel_v: Vec3,
    duration_s: float,
    range_threshold_km: float,
    samples_per_orbit: int = 128,
    steps_per_orbit: int = 128,
) -> list[RendezvousOpportunity]:
    """Predict windows where linear relative range enters a threshold.

    The scan uses the eccentric T/H propagator, not a circular approximation.
    Sampling can miss an interval narrower than the grid spacing; increase
    ``samples_per_orbit`` for short, high-eccentricity encounters.
    """
    if duration_s <= 0.0:
        raise ValueError("duration must be positive")
    if range_threshold_km < 0.0:
        raise ValueError("range threshold must be nonnegative")
    if samples_per_orbit < 8:
        raise ValueError("samples_per_orbit must be at least 8")

    sample_count = max(2, math.ceil(samples_per_orbit * duration_s / chief.period_s))
    dt = duration_s / sample_count
    times = [i * dt for i in range(sample_count + 1)]
    times[-1] = duration_s

    # March sequentially so an N-sample scan costs N short propagations rather
    # than N propagations from epoch.
    ranges: list[float] = []
    scan_chief = chief
    scan_r = rel_r
    scan_v = rel_v
    previous_t = 0.0
    ranges.append(scan_r.norm())
    for t in times[1:]:
        point = eccentric_relative_propagate(
            scan_chief,
            scan_r,
            scan_v,
            t - previous_t,
            steps_per_orbit=steps_per_orbit,
        )
        scan_chief = replace(
            scan_chief, eccentric_anomaly_rad=point.eccentric_anomaly_rad
        )
        scan_r = point.position_km
        scan_v = point.rotating_velocity_km_s
        previous_t = t
        ranges.append(scan_r.norm())

    inside = [value <= range_threshold_km for value in ranges]
    opportunities: list[RendezvousOpportunity] = []
    i = 0
    while i <= sample_count:
        if not inside[i]:
            i += 1
            continue
        start_index = i
        while i + 1 <= sample_count and inside[i + 1]:
            i += 1
        end_index = i

        if start_index == 0:
            start = times[0]
        else:
            start = _bisect_crossing(
                chief,
                rel_r,
                rel_v,
                times[start_index - 1],
                times[start_index],
                range_threshold_km,
                steps_per_orbit,
            )
        if end_index == sample_count:
            end = times[-1]
        else:
            end = _bisect_crossing(
                chief,
                rel_r,
                rel_v,
                times[end_index + 1],
                times[end_index],
                range_threshold_km,
                steps_per_orbit,
            )

        min_index = min(
            range(start_index, end_index + 1), key=lambda index: ranges[index]
        )
        lo = times[max(start_index, min_index - 1)]
        hi = times[min(end_index, min_index + 1)]
        if hi > lo:
            minimum_time, minimum_range = _golden_minimum(
                chief, rel_r, rel_v, lo, hi, steps_per_orbit
            )
        else:
            minimum_time = times[min_index]
            minimum_range = ranges[min_index]

        opportunities.append(
            RendezvousOpportunity(
                start_time_s=start,
                end_time_s=end,
                minimum_time_s=minimum_time,
                minimum_range_km=minimum_range,
            )
        )
        i += 1

    return opportunities


def validate_eccentric_rendezvous(
    chief_r0: Vec3,
    chief_v0: Vec3,
    deputy_r0: Vec3,
    deputy_v0: Vec3,
    commanded_initial_velocity_rtn: Vec3,
    target_relative_position_rtn: Vec3,
    tof_s: float,
    mu: float = MU_EARTH_KM3_S2,
) -> RendezvousValidation:
    """Measure nonlinear two-body miss for a T/H rendezvous burn.

    ``commanded_initial_velocity_rtn`` is the required rotating-frame relative
    velocity returned by :func:`eccentric_targeting`.  The miss quantifies the
    first-order T/H truncation under point-mass gravity; perturbations such as
    J2 and drag are outside this diagnostic.
    """
    chief = EccentricChief.from_state(chief_r0, chief_v0, mu)
    rel_r0, rel_v_inertial0 = relative_hill(
        chief_r0, chief_v0, deputy_r0, deputy_v0
    )
    current_v0 = rotating_relative_velocity(chief, rel_r0, rel_v_inertial0)
    linear_prediction = eccentric_relative_propagate(
        chief, rel_r0, commanded_initial_velocity_rtn, tof_s
    )
    delta_v_rtn = commanded_initial_velocity_rtn - current_v0
    basis = hill_frame(chief_r0, chief_v0)
    delta_v_inertial = (
        basis[0] * delta_v_rtn.x
        + basis[1] * delta_v_rtn.y
        + basis[2] * delta_v_rtn.z
    )

    chief_r1, chief_v1 = propagate_universal(chief_r0, chief_v0, tof_s, mu)
    deputy_r1, deputy_v1 = propagate_universal(
        deputy_r0, deputy_v0 + delta_v_inertial, tof_s, mu
    )
    final_r, final_v = relative_hill(chief_r1, chief_v1, deputy_r1, deputy_v1)
    final_r_error = final_r - target_relative_position_rtn
    final_v_error = final_v - linear_prediction.inertial_velocity_km_s
    return RendezvousValidation(
        position_miss_km=final_r_error.norm(),
        velocity_miss_km_s=final_v_error.norm(),
        final_relative_position_km=final_r,
        final_relative_velocity_km_s=final_v,
    )
