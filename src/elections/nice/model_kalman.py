"""Kalman filter (same model as kalman.py) in the common framework, so it can be
backtested like the others and used in the ensemble.

    offset(h+1) = offset(h) + drift      drift ~ N(0, q)
    reading(h)  = reference(h) + offset(h) + noise    noise ~ N(0, r)

on the log-odds scale; q and r are fitted by maximum likelihood on past elections.
"""

import numpy as np
import pandas as pd

import forecast_common as fc

GRID = np.arange(fc.OPEN + 1, fc.FINAL + 1)
Q_GRID = np.geomspace(1e-5, 5e-2, 20)
R_GRID = np.geomspace(1e-5, 5e-2, 20)


def reference(elections):
    curves = np.array([fc.logit(fc.interp_curve(e, GRID)) for e in elections])
    return pd.Series(np.median(curves, axis=0), index=GRID)


def run_filter(h, v, ref, q, r):
    x, P, t, ll = 0.0, 4.0, h[0], 0.0
    for hh, vv in zip(h, v):
        P += q * (hh - t)
        t = hh
        S = P + r
        resid = fc.logit(vv) - ref[hh] - x
        ll += -0.5 * (np.log(2 * np.pi * S) + resid**2 / S)
        K = P / S
        x, P = x + K * resid, (1 - K) * P
    return ll, x, P, t


def make_model(train):
    refs = {e.name: reference([x for x in train if x.name != e.name]) for e in train}
    data = [(e.h, e.v, refs[e.name]) for e in train]
    q, r = max(
        ((q, r) for q in Q_GRID for r in R_GRID),
        key=lambda p: sum(run_filter(h, v, ref, *p)[0] for h, v, ref in data),
    )
    ref = reference(train)

    def predict(h, v, hours_out):
        _, x, P, t = run_filter(h, v, ref, q, r)
        hours_out = np.asarray(hours_out)
        var = P + q * np.clip(hours_out - t, 0, None)
        mu, sd = ref[hours_out].values + x, np.sqrt(var)
        return pd.DataFrame(
            {
                "lo": fc.expit(mu - fc.Z80 * sd),
                "mid": fc.expit(mu),
                "hi": fc.expit(mu + fc.Z80 * sd),
            },
            index=hours_out,
        )

    predict.params = (q, r)
    return predict


if __name__ == "__main__":
    fc.main(
        make_model,
        "Kalmanov filter",
        "Kalmanov filter na log-odds skali; parametri procenjeni na prošlim izborima.",
        "model_kalman.png",
    )
