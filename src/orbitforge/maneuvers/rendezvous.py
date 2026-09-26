from __future__ import annotations
import math
from orbitforge.core.vector import Vec3
from orbitforge.core.constants import MU_EARTH_KM3_S2
from orbitforge.orbits.relative import th_stm, nu_after_time

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
    vz = n * (target_r.z - c * initial_r.z) / s if abs(s) > 1e-12 else 0.0
    return Vec3(vx, vy, vz)

def _th_required_velocity(initial_r: Vec3, target_r: Vec3, phi):
    # position rows of the TH STM; out-of-plane decouples from the in-plane block
    bx = target_r.x - (phi[0][0] * initial_r.x + phi[0][1] * initial_r.y + phi[0][2] * initial_r.z)
    by = target_r.y - (phi[1][0] * initial_r.x + phi[1][1] * initial_r.y + phi[1][2] * initial_r.z)
    bz = target_r.z - (phi[2][0] * initial_r.x + phi[2][1] * initial_r.y + phi[2][2] * initial_r.z)
    a11, a12 = phi[0][3], phi[0][4]
    a21, a22 = phi[1][3], phi[1][4]
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-12:
        raise ValueError('singular rendezvous transfer')
    vx = (bx * a22 - a12 * by) / det
    vy = (a11 * by - bx * a21) / det
    a33 = phi[2][5]
    if abs(a33) < 1e-12:
        raise ValueError('singular rendezvous transfer')
    return Vec3(vx, vy, bz / a33)

def th_targeting(initial_r: Vec3, target_r: Vec3, a_km: float, e: float, nu0_rad: float, tof_s: float, mu: float = MU_EARTH_KM3_S2):
    """Eccentric-chief analogue of ``cwh_targeting`` (Tschauner-Hempel model).

    Returns the LVLH relative velocity required at t0 so that the linearized
    trajectory reaches target_r after tof_s. Reduces to ``cwh_targeting`` as
    e -> 0. Raises ValueError for singular transfer geometries.
    """
    nu1 = nu_after_time(a_km, e, nu0_rad, tof_s, mu)
    phi = th_stm(a_km, e, nu0_rad, nu1, mu)
    return _th_required_velocity(initial_r, target_r, phi)

def th_rendezvous(initial_r: Vec3, initial_v: Vec3, target_r: Vec3, target_v: Vec3, a_km: float, e: float, nu0_rad: float, tof_s: float, mu: float = MU_EARTH_KM3_S2):
    """Two-impulse rendezvous prediction about an eccentric chief.

    Returns (dv1, dv2) in the LVLH frame: dv1 applied at t0 targets the
    relative state (target_r, target_v) at tof_s, dv2 at arrival nulls the
    remaining relative velocity. Velocities are rotating-frame derivatives.
    """
    nu1 = nu_after_time(a_km, e, nu0_rad, tof_s, mu)
    phi = th_stm(a_km, e, nu0_rad, nu1, mu)
    v_req = _th_required_velocity(initial_r, target_r, phi)
    x0 = [initial_r.x, initial_r.y, initial_r.z, v_req.x, v_req.y, v_req.z]
    vf = [sum(phi[3 + i][j] * x0[j] for j in range(6)) for i in range(3)]
    dv1 = v_req - initial_v
    dv2 = Vec3(target_v.x - vf[0], target_v.y - vf[1], target_v.z - vf[2])
    return (dv1, dv2)
