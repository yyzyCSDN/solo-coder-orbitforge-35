# OrbitForge Mission Lab

OrbitForge is a Python library and small HTTP service for spacecraft mission
analysis: orbital mechanics, mission geometry and resource products.

## Quick start

```bash
python -m pip install -e '.[test]'
PYTHONPATH=src python -m pytest -q
orbitforge
```

The service listens on `127.0.0.1:8080` by default. `GET /live` and `GET /ready`
report service state, and the analysis endpoints live under `/v1/`.

## Eccentric-chief relative motion

`orbitforge.orbits.relative.eccentric_relative_propagate` propagates the
Tschauner--Hempel equations for an elliptic chief and returns the 6x6 state
transition matrix; `eccentric_relative_propagate_from_states` accepts inertial
chief/deputy Cartesian states directly.  Its velocity convention is the time
derivative of the rotating R/T/N position; `rotating_relative_velocity`
converts inertial Cartesian chief/deputy states to that convention.  At
`e = 0` the propagator uses the analytic C/W-Hill matrix, and the eccentric
equations have the same `e -> 0` limit.

`orbitforge.maneuvers.rendezvous.eccentric_targeting` inverts the eccentric
STM for two-point rendezvous, while `predict_rendezvous_opportunities` scans
for threshold-range windows.  `nonlinear_relative_error` compares the linear
prediction with exact two-body propagation.  `instantaneous_circle_proxy_error`
quantifies the separate error made by substituting `sqrt(mu/r^3)` or an
osculating mean motion into the circular C/W-Hill formula; that substitution
is a diagnostic, not eccentric-chief support.
