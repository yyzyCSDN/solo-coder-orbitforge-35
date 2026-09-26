import math

import pytest

from orbitforge.core.constants import MU_EARTH_KM3_S2
from orbitforge.core.vector import Vec3
from orbitforge.maneuvers.rendezvous import (
    cwh_targeting,
    eccentric_targeting,
    predict_rendezvous_opportunities,
    validate_eccentric_rendezvous,
)
from orbitforge.orbits.elements import KeplerianElements, elements_to_state
from orbitforge.orbits.relative import (
    EccentricChief,
    cwh_propagate,
    eccentric_relative_propagate,
    hill_frame,
    inertial_relative_velocity,
    instantaneous_circle_proxy_error,
    nonlinear_relative_error,
)


def _chief(e, a_km=10000.0, nu_rad=0.0):
    elements = KeplerianElements(
        a_km,
        e,
        math.radians(35.0),
        math.radians(40.0),
        math.radians(60.0),
        nu_rad,
    )
    r, v = elements_to_state(elements, MU_EARTH_KM3_S2)
    return EccentricChief.from_state(r, v, MU_EARTH_KM3_S2), r, v


def _deputy_from_rtn(chief, chief_r, chief_v, rel_r, rel_v_rot):
    basis = hill_frame(chief_r, chief_v)
    rel_v_inertial = inertial_relative_velocity(
        chief, chief.eccentric_anomaly_rad, rel_r, rel_v_rot
    )
    dr = basis[0] * rel_r.x + basis[1] * rel_r.y + basis[2] * rel_r.z
    dv = (
        basis[0] * rel_v_inertial.x
        + basis[1] * rel_v_inertial.y
        + basis[2] * rel_v_inertial.z
    )
    return chief_r + dr, chief_v + dv


def test_eccentric_propagation_is_continuous_with_cwh_limit():
    rel_r = Vec3(1.0, -2.0, 0.5)
    rel_v = Vec3(0.01, -0.02, 0.005)
    tof_s = 1234.0

    circular, _, _ = _chief(0.0)
    near_circular, _, _ = _chief(1.0e-11)
    p0 = eccentric_relative_propagate(circular, rel_r, rel_v, tof_s)
    p1 = eccentric_relative_propagate(near_circular, rel_r, rel_v, tof_s)

    assert (p0.position_km - p1.position_km).norm() < 1.0e-9
    assert (p0.rotating_velocity_km_s - p1.rotating_velocity_km_s).norm() < 1.0e-12
    assert (
        p0.position_km
        - cwh_propagate(rel_r, rel_v, circular.mean_motion_rad_s, tof_s)
    ).norm() < 1.0e-12


def test_eccentric_model_matches_two_body_and_exposes_circle_proxy_error():
    chief, chief_r, chief_v = _chief(0.5)
    rel_r = Vec3(0.01, -0.02, 0.005)
    rel_v = Vec3(1.0e-5, -2.0e-5, 3.0e-6)
    deputy_r, deputy_v = _deputy_from_rtn(chief, chief_r, chief_v, rel_r, rel_v)
    tof_s = 0.37 * chief.period_s

    eccentric_error = nonlinear_relative_error(
        chief_r, chief_v, deputy_r, deputy_v, tof_s, steps_per_orbit=512
    )
    circular_proxy = instantaneous_circle_proxy_error(
        chief_r, chief_v, deputy_r, deputy_v, tof_s
    )

    assert eccentric_error.position_error_km < 1.0e-5
    assert eccentric_error.relative_position_error < 1.0e-9
    assert circular_proxy.position_error_km > 1000.0 * eccentric_error.position_error_km


def test_eccentric_targeting_recovers_transfer_and_reports_nonlinear_miss():
    chief, chief_r, chief_v = _chief(0.4)
    rel_r = Vec3(0.05, -0.08, 0.02)
    rel_v = Vec3(2.0e-4, -1.0e-4, 3.0e-5)
    tof_s = 0.22 * chief.period_s
    target = eccentric_relative_propagate(
        chief, rel_r, rel_v, tof_s, steps_per_orbit=512
    ).position_km

    solution = eccentric_targeting(
        chief, rel_r, target, tof_s, current_v=rel_v, steps_per_orbit=512
    )
    assert (solution.initial_velocity_km_s - rel_v).norm() < 1.0e-10
    assert solution.initial_delta_v_km_s is not None
    assert solution.initial_delta_v_km_s.norm() < 1.0e-10

    deputy_r, deputy_v = _deputy_from_rtn(chief, chief_r, chief_v, rel_r, rel_v)
    validation = validate_eccentric_rendezvous(
        chief_r,
        chief_v,
        deputy_r,
        deputy_v,
        solution.initial_velocity_km_s,
        target,
        tof_s,
    )
    assert validation.position_miss_km < 1.0e-4


def test_eccentric_targeting_matches_cwh_targeting_at_zero_eccentricity():
    a_km = 10000.0
    n = math.sqrt(MU_EARTH_KM3_S2 / a_km**3)
    chief = EccentricChief(a_km, 0.0, n, 0.0)
    rel_r = Vec3(1.0, -2.0, 0.5)
    target = Vec3(0.2, 0.3, 0.1)
    tof_s = 1234.0

    circular_v = cwh_targeting(rel_r, target, n, tof_s)
    eccentric_v = eccentric_targeting(chief, rel_r, target, tof_s).initial_velocity_km_s
    assert (circular_v - eccentric_v).norm() < 1.0e-12


def test_rendezvous_opportunity_prediction_uses_eccentric_range():
    chief, _, _ = _chief(0.5)
    rel_r = Vec3(0.2, 0.0, 0.0)
    rel_v = Vec3(0.0, 0.0, 0.0)

    opportunities = predict_rendezvous_opportunities(
        chief,
        rel_r,
        rel_v,
        duration_s=chief.period_s,
        range_threshold_km=0.25,
        samples_per_orbit=128,
        steps_per_orbit=64,
    )

    assert opportunities
    assert opportunities[0].start_time_s == pytest.approx(0.0)
    assert opportunities[0].minimum_range_km <= 0.25
