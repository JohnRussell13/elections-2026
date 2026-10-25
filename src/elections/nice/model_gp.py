"""Model 2: Gaussian process over the turnout curve (log-odds scale).

    logit turnout_e(t) = m(t) + g_e(t) + noise
    m(t)   = median curve of past elections
    g_e(t) = how election e differs from the median, a Gaussian process with
             kernel  c² (a level shift for the whole day)
                   + a² · exp(-(s-t)² / 2ℓ²)  (deviations that last ~ℓ hours)
    noise  = reading noise (different sources, rounding)

c, a, ℓ and the noise are fitted by maximising the marginal likelihood of the
past elections. The forecast is the GP conditioned on today's readings.
The Kalman filter is a special case of this (one particular kernel); here the
data chooses how long deviations last and how much of them is a fixed level.
"""

import itertools

import numpy as np
import pandas as pd

import forecast_common as fc

GRID = np.arange(fc.OPEN + 1, fc.FINAL + 1)  # 8h..20h
NU = 3  # Student-t process: the band widens when today is far from anything seen
# before (like 2026). None = plain Gaussian process.

C_GRID = np.geomspace(0.05, 1.5, 9)
A_GRID = np.geomspace(0.02, 1.0, 9)
L_GRID = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0])
N_GRID = np.geomspace(0.005, 0.1, 6)


def mean_curve(train):
    curves = np.array([fc.logit(fc.interp_curve(e, GRID)) for e in train])
    return pd.Series(np.median(curves, axis=0), index=GRID)


def kernel(s, t, c, a, ell):
    s, t = np.asarray(s, float)[:, None], np.asarray(t, float)[None, :]
    return c**2 + a**2 * np.exp(-((s - t) ** 2) / (2 * ell**2))


def log_marginal(train, means, c, a, ell, noise):
    """Each election is compared with the median curve of the *other* elections,
    otherwise the median partly fits it already and the deviations look too small."""
    total = 0.0
    for e in train:
        y = fc.logit(e.v) - means[e.name][e.h].values
        K = kernel(e.h, e.h, c, a, ell) + noise**2 * np.eye(len(e.h))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L, y)
        total += (
            -0.5 * alpha @ alpha
            - np.log(np.diag(L)).sum()
            - 0.5 * len(y) * np.log(2 * np.pi)
        )
    return total


def make_model(train):
    m = mean_curve(train)
    means = {e.name: mean_curve([x for x in train if x.name != e.name]) for e in train}
    best = max(
        itertools.product(C_GRID, A_GRID, L_GRID, N_GRID),
        key=lambda p: log_marginal(train, means, *p),
    )
    c, a, ell, noise = best

    def predict(h, v, hours_out):
        y = fc.logit(v) - m[h].values
        K = kernel(h, h, c, a, ell) + noise**2 * np.eye(len(h))
        Ks = kernel(hours_out, h, c, a, ell)
        Kss = kernel(hours_out, hours_out, c, a, ell)
        sol = np.linalg.solve(K, np.c_[y, Ks.T])
        mu = m[hours_out].values + Ks @ sol[:, 0]
        var = np.clip(np.diag(Kss - Ks @ sol[:, 1:]), 1e-12, None)
        if NU:
            beta = y @ sol[:, 0]  # how unusual today is so far
            var = var * (NU + beta) / (NU + len(h) - 2)
        sd = np.sqrt(var)
        return pd.DataFrame(
            {
                "lo": fc.expit(mu - fc.Z80 * sd),
                "mid": fc.expit(mu),
                "hi": fc.expit(mu + fc.Z80 * sd),
            },
            index=hours_out,
        )

    predict.params = dict(c=c, a=a, ell=ell, noise=noise)
    return predict


def report(predict, target):
    p = predict.params
    print(
        f"\nNaučeni parametri: pomak nivoa c={p['c']:.3f}, odstupanja a={p['a']:.3f} "
        f"u trajanju ℓ={p['ell']:.0f}h, šum merenja={p['noise']:.3f} (log-odds)"
    )


if __name__ == "__main__":
    fc.main(
        make_model,
        "Gausov / Student-t proces",
        "Student-t proces na log-odds skali: oblik dana i trajanje odstupanja "
        "naučeni iz prošlih izbora; opseg se širi kad je dan neobičan.",
        "model_gp.png",
        report=report,
    )
