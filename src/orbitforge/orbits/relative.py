"""Relative motion in the chief's LVLH (Hill) frame.

Circular chief: Clohessy-Wiltshire-Hill model (``cwh_*``).

Eccentric chief: Tschauner-Hempel model (``th_*``) — the exact linearization of
two-body relative motion about an eccentric Keplerian orbit, solved in closed
form with true anomaly as the independent variable and scaled coordinates
X = (1 + e cos nu) x. The fundamental solutions used here are regular for
0 <= e < 1 (Yamanaka-Ankersen style), so no in-plane singularities appear at
nu = 0 or pi. As e -> 0 the TH state transition matrix reduces exactly to the
CWH matrix, so eccentric results connect continuously to the circular ones
(this is verified by tests, not assumed).

Note: substituting the instantaneous mean motion n(t) = sqrt(mu / r(t)^3) into
the CWH formulas is NOT a valid eccentric model. CWH assumes constant n; with a
time-varying n the in-plane coupling and along-track drift terms are wrong at
O(e) and the error grows secularly. The TH state transition matrix is the
correct linear map and is what ``th_*`` uses.

Error budget (see ``linearization_error_estimate``):
- dominant term: the neglected O((rho / r_c)^2) gravity-gradient nonlinearity;
  the along-track position error grows roughly as (3 mu / r_c^4) rho^2 t^2 / 2
  (order of magnitude), radial/cross-track errors stay bounded per orbit;
- for e < 1e-7 one drift-basis element is evaluated at its analytic e -> 0
  limit, contributing an O(e) model error;
- everything else is evaluated in closed form at roundoff level.
"""
from __future__ import annotations
import math
from orbitforge.core.vector import Vec3
from orbitforge.core.constants import MU_EARTH_KM3_S2
from orbitforge.orbits.kepler import solve_kepler_elliptic, true_from_eccentric, eccentric_from_true

def hill_frame(chief_r: Vec3, chief_v: Vec3):
    x = chief_r.unit()
    z = chief_r.cross(chief_v).unit()
    y = z.cross(x)
    return (x, y, z)

def relative_hill(chief_r, chief_v, deputy_r, deputy_v):
    x, y, z = hill_frame(chief_r, chief_v)
    dr = deputy_r - chief_r
    dv = deputy_v - chief_v
    return (Vec3(dr.dot(x), dr.dot(y), dr.dot(z)), Vec3(dv.dot(x), dv.dot(y), dv.dot(z)))

def relative_state(chief_r, chief_v, deputy_r, deputy_v):
    """LVLH relative position and rotating-frame relative velocity.

    Unlike ``relative_hill`` (which only projects the inertial velocity
    difference), the velocity returned here is the time derivative in the
    rotating Hill frame, i.e. the convention the CWH/TH dynamics use.
    """
    x, y, z = hill_frame(chief_r, chief_v)
    dr = deputy_r - chief_r
    dv = deputy_v - chief_v
    w = chief_r.cross(chief_v).norm() / chief_r.norm2()
    rl = Vec3(dr.dot(x), dr.dot(y), dr.dot(z))
    vi = Vec3(dv.dot(x), dv.dot(y), dv.dot(z))
    return (rl, Vec3(vi.x + w * rl.y, vi.y - w * rl.x, vi.z))

def absolute_state(chief_r, chief_v, rel_r, rel_v):
    """Inverse of ``relative_state``: recover the deputy's inertial state."""
    x, y, z = hill_frame(chief_r, chief_v)
    w = chief_r.cross(chief_v).norm() / chief_r.norm2()
    dr = x * rel_r.x + y * rel_r.y + z * rel_r.z
    vi = Vec3(rel_v.x - w * rel_r.y, rel_v.y + w * rel_r.x, rel_v.z)
    dv = x * vi.x + y * vi.y + z * vi.z
    return (chief_r + dr, chief_v + dv)

def cwh_propagate(rel_r: Vec3, rel_v: Vec3, n: float, t: float):
    c, s = (math.cos(n * t), math.sin(n * t))
    x = (4 - 3 * c) * rel_r.x + s / n * rel_v.x + 2 * (1 - c) / n * rel_v.y
    y = 6 * (s - n * t) * rel_r.x + rel_r.y - 2 * (1 - c) / n * rel_v.x + (4 * s - 3 * n * t) / n * rel_v.y
    z = c * rel_r.z + s / n * rel_v.z
    return Vec3(x, y, z)

def cwh_propagate_full(rel_r: Vec3, rel_v: Vec3, n: float, t: float):
    """CWH propagation of position and velocity (``cwh_propagate`` is position-only)."""
    nt = n * t
    c, s = (math.cos(nt), math.sin(nt))
    r = cwh_propagate(rel_r, rel_v, n, t)
    vx = 3 * n * s * rel_r.x + c * rel_v.x + 2 * s * rel_v.y
    vy = 6 * n * (c - 1) * rel_r.x - 2 * s * rel_v.x + (4 * c - 3) * rel_v.y
    vz = -n * s * rel_r.z + c * rel_v.z
    return (r, Vec3(vx, vy, vz))

# --- Tschauner-Hempel model (eccentric chief) ---

# Below this eccentricity the drift basis element Y_w is evaluated at its
# analytic e -> 0 limit; the induced model error is O(e) ~ 1e-7.
_E_LIMIT = 1e-7

def _unwrap(wrapped: float, reference: float) -> float:
    return wrapped + 2.0 * math.pi * round((reference - wrapped) / (2.0 * math.pi))

def _eccentric_anomaly(nu: float, e: float) -> float:
    # unwrapped to the same branch as the (possibly multi-revolution) nu
    return _unwrap(eccentric_from_true(nu, e), nu)

def nu_after_time(a_km: float, e: float, nu0_rad: float, tof_s: float, mu: float = MU_EARTH_KM3_S2) -> float:
    """True anomaly reached tof_s after the chief was at nu0_rad (unwrapped)."""
    n = math.sqrt(mu / a_km ** 3)
    E0 = _eccentric_anomaly(nu0_rad, e)
    M1 = E0 - e * math.sin(E0) + n * tof_s
    E1 = _unwrap(solve_kepler_elliptic(M1, e), M1)
    return _unwrap(true_from_eccentric(E1, e), E1)

def _th_fundamental(nu: float, e: float):
    """G(nu): columns are six independent TH solutions in scaled coordinates.

    State order (X, Y, Z, X', Y', Z') with X = (1 + e cos nu) x etc. and
    primes d/dnu. In-plane columns: s = rho sin(nu), c = rho cos(nu), the
    drift solution w = s K + cos^2(nu)(1+rho)/(2 rho) with
    K = integral cos(nu)/rho^3 dnu (closed form in eccentric anomaly), and the
    constant along-track offset. Out-of-plane: cos(nu), sin(nu).
    """
    cn, sn = math.cos(nu), math.sin(nu)
    rho = 1.0 + e * cn
    s, c = rho * sn, rho * cn
    sp = cn + e * math.cos(2.0 * nu)
    cp = -sn - e * math.sin(2.0 * nu)
    E = _eccentric_anomaly(nu, e)
    K = ((1.0 + e * e) * math.sin(E) - 1.5 * e * E - 0.25 * e * math.sin(2.0 * E)) / (1.0 - e * e) ** 2.5
    sig = -cn + 0.5 * e * sn * sn
    w = s * K + cn * cn * (1.0 + rho) / (2.0 * rho)
    wp = sp * K + s * cn / rho ** 3 - sn * cn * (1.0 + 1.0 / rho) + 0.5 * e * sn * cn * cn / (rho * rho)
    ys = 2.0 * cn - e * sn * sn
    yc = -(sn + s)
    if e >= _E_LIMIT:
        yw = (2.0 * K * (1.0 + e * e - 2.0 * e * sig) - (sn + s)) / (2.0 * e)
    else:
        yw = -1.5 * nu
    return [
        [s, c, w, 0.0, 0.0, 0.0],
        [ys, yc, yw, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, cn, sn],
        [sp, cp, wp, 0.0, 0.0, 0.0],
        [-2.0 * s, e - 2.0 * c, 0.5 - 2.0 * w, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -sn, cn],
    ]

def _scaled_to_physical(nu: float, e: float, nbar: float):
    rho = 1.0 + e * math.cos(nu)
    rhop = -e * math.sin(nu)
    d = [[0.0] * 6 for _ in range(6)]
    for i in range(3):
        d[i][i] = 1.0 / rho
        d[3 + i][i] = -nbar * rhop
        d[3 + i][3 + i] = nbar * rho
    return d

def _physical_to_scaled(nu: float, e: float, nbar: float):
    rho = 1.0 + e * math.cos(nu)
    rhop = -e * math.sin(nu)
    d = [[0.0] * 6 for _ in range(6)]
    for i in range(3):
        d[i][i] = rho
        d[3 + i][i] = rhop
        d[3 + i][3 + i] = 1.0 / (nbar * rho)
    return d

def _mat_mul(a, b):
    n, m, p = len(a), len(b[0]), len(b)
    return [[sum(a[i][k] * b[k][j] for k in range(p)) for j in range(m)] for i in range(n)]

def _mat_vec(a, x):
    return [sum(row[j] * x[j] for j in range(len(x))) for row in a]

def _mat_inv(m):
    n = len(m)
    a = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(m)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(a[r][i]))
        if abs(a[p][i]) < 1e-300:
            raise ValueError('singular matrix')
        a[i], a[p] = a[p], a[i]
        d = a[i][i]
        a[i] = [v / d for v in a[i]]
        for r in range(n):
            if r != i:
                f = a[r][i]
                if f != 0.0:
                    a[r] = [v - f * w for v, w in zip(a[r], a[i])]
    return [row[n:] for row in a]

def th_stm(a_km: float, e: float, nu0_rad: float, nu1_rad: float, mu: float = MU_EARTH_KM3_S2):
    """State transition matrix of the TH model from nu0 to nu1 (6x6, row-major).

    Maps (x, y, z, vx, vy, vz) in the chief LVLH frame at nu0 to the same
    quantities at nu1; velocities are rotating-frame time derivatives (km/s).
    """
    if not 0.0 <= e < 1.0:
        raise ValueError('elliptic eccentricity required')
    if a_km <= 0.0:
        raise ValueError('positive semimajor axis required')
    p = a_km * (1.0 - e * e)
    nbar = math.sqrt(mu / p ** 3)
    g1 = _th_fundamental(nu1_rad, e)
    g0_inv = _mat_inv(_th_fundamental(nu0_rad, e))
    d1 = _scaled_to_physical(nu1_rad, e, nbar)
    d0_inv = _physical_to_scaled(nu0_rad, e, nbar)
    return _mat_mul(d1, _mat_mul(g1, _mat_mul(g0_inv, d0_inv)))

def th_propagate(rel_r: Vec3, rel_v: Vec3, a_km: float, e: float, nu0_rad: float, tof_s: float, mu: float = MU_EARTH_KM3_S2):
    """Propagate an LVLH relative state about an eccentric chief by tof_s.

    rel_v is the rotating-frame relative velocity (see ``relative_state``).
    Returns (position km, velocity km/s) at chief true anomaly nu0 + Delta nu.
    """
    nu1 = nu_after_time(a_km, e, nu0_rad, tof_s, mu)
    phi = th_stm(a_km, e, nu0_rad, nu1, mu)
    x0 = [rel_r.x, rel_r.y, rel_r.z, rel_v.x, rel_v.y, rel_v.z]
    x1 = _mat_vec(phi, x0)
    return (Vec3(x1[0], x1[1], x1[2]), Vec3(x1[3], x1[4], x1[5]))

def linearization_error_estimate(a_km: float, e: float, nu_rad: float, rel_r: Vec3, tof_s: float, mu: float = MU_EARTH_KM3_S2) -> float:
    """Order-of-magnitude position error (km) of the linearized eccentric model.

    The TH/CWH models drop the quadratic-and-higher terms of the gravity
    gradient, whose characteristic acceleration is 3 mu rho^2 / r_c^4 with
    rho = |rel_r| and r_c the chief radius at nu_rad. The estimate below is
    the along-track-dominated accumulation of that acceleration over tof_s;
    it scales like rho^2 and t^2, which is the scaling tests verify against
    the full nonlinear two-body dynamics.
    """
    r_c = a_km * (1.0 - e * e) / (1.0 + e * math.cos(nu_rad))
    rho = rel_r.norm()
    acc2 = 3.0 * mu * rho * rho / r_c ** 4
    return 0.5 * acc2 * tof_s * tof_s
