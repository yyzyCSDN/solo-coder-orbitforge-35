import math
import pytest
from orbitforge.core.vector import Vec3
from orbitforge.core.state import CartesianState
from orbitforge.core.constants import MU_EARTH_KM3_S2 as MU
from orbitforge.orbits.elements import KeplerianElements, elements_to_state
from orbitforge.orbits.two_body import propagate_two_body
from orbitforge.orbits.relative import (
    cwh_propagate_full, th_propagate, th_stm, relative_state, absolute_state,
    linearization_error_estimate, nu_after_time, _th_fundamental)
from orbitforge.maneuvers.rendezvous import cwh_targeting, th_targeting, th_rendezvous

A = 12000.0
T = 2.0 * math.pi * math.sqrt(A ** 3 / MU)
N = math.sqrt(MU / A ** 3)
R0 = Vec3(0.4, -0.6, 0.3)
V0 = Vec3(2e-4, -3e-4, 1e-4)

def _truth(e, nu0, rel_r, rel_v, tof, i=0.4):
    el = KeplerianElements(A, e, i, 0.5, 0.6, nu0)
    rc0, vc0 = elements_to_state(el)
    rd0, vd0 = absolute_state(rc0, vc0, rel_r, rel_v)
    sc = propagate_two_body(CartesianState(0.0, rc0, vc0), tof)
    sd = propagate_two_body(CartesianState(0.0, rd0, vd0), tof)
    return relative_state(sc.position_km, sc.velocity_km_s, sd.position_km, sd.velocity_km_s)

def test_relative_state_roundtrip():
    el = KeplerianElements(A, 0.3, 0.4, 0.5, 0.6, 1.2)
    rc, vc = elements_to_state(el)
    rd, vd = absolute_state(rc, vc, R0, V0)
    r2, v2 = relative_state(rc, vc, rd, vd)
    assert (r2 - R0).norm() < 1e-12 and (v2 - V0).norm() < 1e-15

def test_th_reduces_to_cwh_at_zero_eccentricity():
    r1, v1 = th_propagate(R0, V0, A, 0.0, 0.7, 0.8 * T)
    r2, v2 = cwh_propagate_full(R0, V0, N, 0.8 * T)
    assert (r1 - r2).norm() < 1e-12
    assert (v1 - v2).norm() < 1e-15

def test_th_continuous_as_eccentricity_vanishes():
    # the eccentric model must approach the circular one linearly in e;
    # this fails if someone just feeds osculating n(t) into the CWH formulas
    rc, vc = cwh_propagate_full(R0, V0, N, 0.8 * T)
    diffs = []
    for e in (1e-2, 1e-4, 1e-6, 1e-8):
        r1, _ = th_propagate(R0, V0, A, e, 0.7, 0.8 * T)
        diffs.append((r1 - rc).norm())
    assert diffs[1] / diffs[0] < 0.05 and diffs[2] / diffs[1] < 0.05
    assert diffs[3] < 1e-6

def test_th_matches_nonlinear_two_body():
    for e in (0.1, 0.3, 0.5):
        tof = 1.7 * T
        r1, _ = th_propagate(R0, V0, A, e, 0.7, tof)
        rt, _ = _truth(e, 0.7, R0, V0, tof)
        err = (r1 - rt).norm()
        est = linearization_error_estimate(A, e, 0.7, R0, tof)
        assert err < 5.0 * est and est < 5.0 * err

def test_th_error_scales_quadratically_with_offset():
    # doubling the relative offset must quadruple the linearization error:
    # the residual is the O(rho^2) gravity-gradient term, not an artifact
    e, tof = 0.3, 1.7 * T
    errs = []
    for k in (1.0, 2.0, 4.0):
        rk, vk = R0 * k, V0 * k
        r1, _ = th_propagate(rk, vk, A, e, 0.7, tof)
        rt, _ = _truth(e, 0.7, rk, vk, tof)
        errs.append((r1 - rt).norm())
    assert 3.0 < errs[1] / errs[0] < 5.0
    assert 3.0 < errs[2] / errs[1] < 5.0

def test_th_multi_orbit_and_backward():
    r1, _ = th_propagate(R0, V0, A, 0.4, 0.7, 3.4 * T)
    rt, _ = _truth(0.4, 0.7, R0, V0, 3.4 * T)
    assert (r1 - rt).norm() < 1.0
    r1, _ = th_propagate(R0, V0, A, 0.4, 0.7, -0.6 * T)
    rt, _ = _truth(0.4, 0.7, R0, V0, -0.6 * T)
    assert (r1 - rt).norm() < 0.05

def test_th_stm_group_property():
    for e in (0.0, 0.3, 0.7):
        p02 = th_stm(A, e, 0.4, 7.7)
        p01 = th_stm(A, e, 0.4, 2.9)
        p12 = th_stm(A, e, 2.9, 7.7)
        for i in range(6):
            for j in range(6):
                s = sum(p12[i][k] * p01[k][j] for k in range(6))
                assert abs(s - p02[i][j]) < 1e-8

def test_th_fundamental_solutions_satisfy_ode():
    # every column of G must solve X'' - 2Y' - 3X/rho = 0, Y'' + 2X' = 0, Z'' + Z = 0
    for e in (0.0, 0.3, 0.6, 0.9):
        for nu in (0.05, 1.4, 2.9, 4.6, 6.1, 20.5):
            h = 1e-5
            gp, g0, gm = (_th_fundamental(nu + h, e), _th_fundamental(nu, e), _th_fundamental(nu - h, e))
            rho = 1.0 + e * math.cos(nu)
            for j in range(4):
                dx = (gp[0][j] - gm[0][j]) / (2 * h)
                dy = (gp[1][j] - gm[1][j]) / (2 * h)
                dxp = (gp[3][j] - gm[3][j]) / (2 * h)
                assert abs(dx - g0[3][j]) < 1e-6
                assert abs(dy - g0[4][j]) < 1e-6
                assert abs(dxp - (2.0 * g0[4][j] + 3.0 * g0[0][j] / rho)) < 1e-6
            dzp = (gp[5][4] - gm[5][4]) / (2 * h)
            assert abs(dzp + g0[2][4]) < 1e-9

def test_nu_after_time_one_period():
    for e in (0.0, 0.5):
        assert abs(nu_after_time(A, e, 0.7, T) - 0.7 - 2.0 * math.pi) < 1e-9

def test_th_targeting_hits_target():
    e, nu0, tof = 0.25, 1.1, 0.9 * T
    tgt = Vec3(0.1, -0.2, 0.05)
    v_req = th_targeting(R0, tgt, A, e, nu0, tof)
    r1, _ = th_propagate(R0, v_req, A, e, nu0, tof)
    assert (r1 - tgt).norm() < 1e-9

def test_th_targeting_continuous_with_cwh():
    tgt = Vec3(0.1, -0.2, 0.05)
    vc = cwh_targeting(R0, tgt, N, 0.9 * T)
    diffs = []
    for e in (1e-3, 1e-5):
        vq = th_targeting(R0, tgt, A, e, 1.1, 0.9 * T)
        diffs.append((vq - vc).norm())
    assert diffs[1] / diffs[0] < 0.05 and diffs[1] < 1e-8

def test_th_targeting_singular_transfer():
    with pytest.raises(ValueError):
        th_targeting(R0, Vec3(0.0, 0.0, 0.0), A, 0.0, 0.3, T)

def test_th_rendezvous_two_impulse_against_nonlinear():
    e, nu0, tof = 0.2, 0.9, 1.2 * T
    dv1, dv2 = th_rendezvous(R0, V0, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), A, e, nu0, tof)
    el = KeplerianElements(A, e, 0.4, 0.5, 0.6, nu0)
    rc0, vc0 = elements_to_state(el)
    rd0, vd0 = absolute_state(rc0, vc0, R0, V0 + dv1)
    sc = propagate_two_body(CartesianState(0.0, rc0, vc0), tof)
    sd = propagate_two_body(CartesianState(0.0, rd0, vd0), tof)
    miss = (sd.position_km - sc.position_km).norm()
    est = linearization_error_estimate(A, e, nu0, R0, tof)
    assert miss < 2.0 * est
