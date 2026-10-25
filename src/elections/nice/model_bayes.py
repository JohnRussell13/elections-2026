"""Model 4: hierarchical Bayesian model.

Same curve as model 1 (turnout = K × Weibull timing), but every election gets
its own three parameters

    θ_e = (logit K, log λ, log k)   ~   Normal(μ, Σ)      (population of elections)

and the population mean μ and covariance Σ are themselves uncertain
(Normal–Inverse-Wishart prior). With 11 past elections that uncertainty matters,
and it turns the prior for a new election into a multivariate Student-t
(the exact posterior predictive), so an unusual day is allowed.

Two differences from model 1:
  * K and timing are modelled jointly, so if past high-turnout days also voted
    earlier, the model uses that link;
  * the uncertainty about the population itself (only 11 days) is included.

The past elections enter through their fitted θ_e (fit noise is small next to
the spread between elections). Today's posterior is computed exactly on a grid.
"""

import numpy as np
import pandas as pd
from math import lgamma

import forecast_common as fc
import model_saturation as ms

N_SAMPLES = 40000
SEED = 0
KAPPA0 = 0.1  # weight of the prior mean (in "elections")
NU0 = 4  # prior degrees of freedom for Σ (minimum is 3 + 1)
PSI0_SD = (0.3, 0.3, 0.3)  # prior guess of the spread of logit K, log λ, log k

LOGIT_K = np.linspace(
    float(fc.logit(30)), float(fc.logit(95)), 120
)  # K from 30% to 95%


def niw_predictive(theta):
    """Multivariate t predictive for a new election from the fitted θ_e."""
    n, d = theta.shape
    mu0 = theta.mean(axis=0)
    psi0 = np.diag(np.square(PSI0_SD)) * (NU0 - d - 1)
    xbar = theta.mean(axis=0)
    S = (theta - xbar).T @ (theta - xbar)
    kn, nun = KAPPA0 + n, NU0 + n
    mun = (KAPPA0 * mu0 + n * xbar) / kn
    psin = psi0 + S + KAPPA0 * n / kn * np.outer(xbar - mu0, xbar - mu0)
    df = nun - d + 1
    scale = psin * (kn + 1) / (kn * df)
    return mun, scale, df


def mvt_logpdf(x, mu, scale, df):
    d = len(mu)
    inv = np.linalg.inv(scale)
    diff = x - mu
    maha = np.einsum("ij,jk,ik->i", diff, inv, diff)
    return (
        lgamma((df + d) / 2)
        - lgamma(df / 2)
        - d / 2 * np.log(df * np.pi)
        - 0.5 * np.log(np.linalg.det(scale))
        - (df + d) / 2 * np.log1p(maha / df)
    )


def make_model(train):
    fits = [ms.fit_one(e.h, e.v) for e in train]
    theta = np.array([[fc.logit(K), gl, gk] for K, gl, gk, _ in fits])
    n_obs = sum(len(e.h) for e in train)
    sigma = np.sqrt(sum(f[3] for f in fits) / max(n_obs - 3 * len(train), 1))
    mu, scale, df = niw_predictive(theta)

    # Full 3-D grid: timing grid from model 1 × final-turnout grid
    lk = LOGIT_K[None, :]
    K_vals = fc.expit(lk)  # (1, nK)
    pts = np.c_[
        np.repeat(LOGIT_K, len(ms.GL)),
        np.tile(ms.GL, len(LOGIT_K)),
        np.tile(ms.GK, len(LOGIT_K)),
    ]
    log_prior = mvt_logpdf(pts, mu, scale, df).reshape(len(LOGIT_K), len(ms.GL)).T

    def posterior(h, v):
        Fo = ms.F(h)  # (nG, nobs)
        pred = K_vals[..., None] * Fo[:, None, :]  # (nG, nK, nobs)
        rss = ((v[None, None, :] - pred) ** 2).sum(axis=2)
        logw = log_prior - rss / (2 * sigma**2)
        w = np.exp(logw - logw.max())
        return w / w.sum()

    def predict(h, v, hours_out):
        w = posterior(h, v).ravel()
        rng = np.random.default_rng(SEED)
        idx = rng.choice(w.size, size=N_SAMPLES, p=w)
        gi, ki = np.unravel_index(idx, (len(ms.GL), len(LOGIT_K)))
        # spread K within its grid cell so quantiles are smooth
        step = LOGIT_K[1] - LOGIT_K[0]
        K = fc.expit(LOGIT_K[ki] + rng.uniform(-step / 2, step / 2, N_SAMPLES))
        draws = K[:, None] * ms.F(hours_out, ms.GL[gi], ms.GK[gi])
        q = np.quantile(draws, [0.1, 0.5, 0.9], axis=0)
        return pd.DataFrame(q.T, index=hours_out, columns=["lo", "mid", "hi"])

    predict.population = (mu, scale, df)
    predict.theta = theta
    return predict


def report(predict, target):
    mu, scale, df = predict.population
    sd = np.sqrt(np.diag(scale))
    corr = scale / np.outer(sd, sd)
    print(f"\nPopulacija izbora (prediktivna t raspodela, df={df}):")
    print(f"  tipična konačna izlaznost {fc.expit(mu[0]):.1f}%")
    print(
        f"  korelacija: konačna izlaznost ↔ rano glasanje (−log λ): {-corr[0, 1]:+.2f}"
    )


if __name__ == "__main__":
    fc.main(
        make_model,
        "hijerarhijski Bayes",
        "Hijerarhijski Bayes: konačna izlaznost i ritam dolaska zajedno, "
        "uz nesigurnost o tome kakvi su izbori uopšte (samo 11 dana).",
        "model_bayes.png",
        report=report,
    )
