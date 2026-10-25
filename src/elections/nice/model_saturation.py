"""Model 1: saturation curve, separating "how many vote" from "how early they vote".

    turnout(t) = K × F(t)

  K  = final turnout (share of registered voters who vote by 20h)
  F  = share of those voters who have already voted by hour t (0 at 7h, 1 at 20h),
       a Weibull curve with a scale (λ: how early) and a shape (k: how bunched)

Each past election is fitted on its own, which gives the usual range of timing
(λ, k). For the current election the timing gets that range as a prior, K gets a
flat prior, and the posterior is computed over a grid, so there is no hand tuning:
the noise level comes from how well the curve fits past elections.

This answers the question the other models cannot: is turnout high because more
people vote (high K) or because the same people vote earlier (shifted F)?
"""

import numpy as np
import pandas as pd

import forecast_common as fc

PRIOR_SCALE = 1.0  # >1 widens the timing prior
PRIOR_DF = 3  # Student-t prior: heavy tails let a day with unusual timing (like
# 2026) override the past instead of being pulled back to it.
# Set to None for a normal prior.
K_PRIOR_WIDTH = 2.0  # prior on final turnout: Student-t around past finals, scale =
# this × their spread (wide, so 2026 can be far from the past)
N_SAMPLES = 40000
SEED = 0

S_END = fc.FINAL - fc.OPEN
LOG_LAM = np.linspace(np.log(2.0), np.log(80.0), 90)
LOG_K = np.linspace(np.log(0.5), np.log(4.0), 70)
GL, GK = np.meshgrid(LOG_LAM, LOG_K, indexing="ij")
GL, GK = GL.ravel(), GK.ravel()


def F(hours, log_lam=GL, log_k=GK):
    """Timing curve for every grid point: array (n_grid, n_hours)."""
    s = np.asarray(hours, float) - fc.OPEN
    lam, k = np.exp(log_lam)[:, None], np.exp(log_k)[:, None]
    cdf = lambda x: 1 - np.exp(-((x / lam) ** k))
    return cdf(s[None, :]) / cdf(np.array([[S_END]]))


def fit_one(h, v):
    """Least-squares K, λ, k for one complete past election."""
    Fo = F(h)
    K = (Fo @ v) / (Fo**2).sum(axis=1)
    rss = ((v[None, :] - K[:, None] * Fo) ** 2).sum(axis=1)
    i = rss.argmin()
    return K[i], GL[i], GK[i], rss[i]


def make_model(train):
    fits = [fit_one(e.h, e.v) for e in train]
    timing = np.array([[f[1], f[2]] for f in fits])
    mean = timing.mean(axis=0)
    cov = np.cov(timing.T) * PRIOR_SCALE**2 + np.eye(2) * 1e-3
    inv, logdet = np.linalg.inv(cov), np.log(np.linalg.det(cov))
    n_obs = sum(len(e.h) for e in train)
    sigma = np.sqrt(sum(f[3] for f in fits) / max(n_obs - 3 * len(train), 1))
    d = np.c_[GL, GK] - mean
    maha = np.einsum("ij,jk,ik->i", d, inv, d)
    if PRIOR_DF:
        log_prior = -(PRIOR_DF + 2) / 2 * np.log1p(maha / PRIOR_DF) - 0.5 * logdet
    else:
        log_prior = -0.5 * maha - 0.5 * logdet

    def posterior(h, v):
        Fo = F(h)
        sff = (Fo**2).sum(axis=1)
        K_hat = (Fo @ v) / sff
        rss = ((v[None, :] - K_hat[:, None] * Fo) ** 2).sum(axis=1)
        logw = log_prior - rss / (2 * sigma**2) - 0.5 * np.log(sff)
        w = np.exp(logw - logw.max())
        return w / w.sum(), K_hat, sigma / np.sqrt(sff)

    finals = np.array([e.final for e in train])
    k_mid, k_scale = finals.mean(), finals.std(ddof=1) * K_PRIOR_WIDTH

    def sample(h, v):
        """Posterior draws of (timing grid index, K), with the prior on K applied."""
        w, K_hat, K_sd = posterior(h, v)
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(w), size=N_SAMPLES, p=w)
        K = K_hat[idx] + K_sd[idx] * rng.standard_normal(N_SAMPLES)
        kw = (1 + ((K - k_mid) / k_scale) ** 2 / 3) ** -2 * (K > 0) * (K <= 100)
        keep = rng.choice(N_SAMPLES, size=N_SAMPLES, p=kw / kw.sum())
        return idx[keep], K[keep]

    def predict(h, v, hours_out):
        idx, K = sample(h, v)
        Fh = F(hours_out, GL[idx], GK[idx])
        draws = np.clip(K[:, None] * Fh, 0, 100)
        q = np.quantile(draws, [0.1, 0.5, 0.9], axis=0)
        return pd.DataFrame(q.T, index=hours_out, columns=["lo", "mid", "hi"])

    predict.sample = sample
    predict.prior_mean = mean
    predict.sigma = sigma
    return predict


def share_by(hour, log_lam, log_k):
    return F([hour], np.atleast_1d(log_lam), np.atleast_1d(log_k))[:, 0]


def report(predict, target):
    """How many voters vs how early: compare 2026 timing with past elections."""
    idx, K = predict.sample(target.h, target.v)
    K_mean = K.mean()
    print(
        f"\nKoliko birača vs. koliko rano (šum merenja σ = {predict.sigma:.2f} p.p.):"
    )
    print(f"  Procenjena konačna izlaznost K: {K_mean:.1f}%")
    for hour in [10, 12, 14, 16, 18]:
        now = share_by(hour, GL[idx], GK[idx]).mean() * 100
        usual = share_by(hour, *predict.prior_mean)[0] * 100
        print(
            f"  do {hour}h glasalo {now:4.1f}% od konačnog broja birača "
            f"(uobičajeno {usual:4.1f}%)"
        )


def extra(predict, target):
    """Dotted line: same final turnout K, but voting at the usual pace."""
    K_mean = predict.sample(target.h, target.v)[1].mean()
    hrs = np.arange(fc.OPEN, fc.FINAL + 1)
    usual = (
        K_mean
        * F(hrs, np.array([predict.prior_mean[0]]), np.array([predict.prior_mean[1]]))[
            0
        ]
    )

    def draw(ax):
        import chart_style as cs

        ax.plot(
            hrs,
            usual,
            color=cs.PAST,
            lw=1.6,
            ls=":",
            zorder=3,
            label=f"isti broj birača ({K_mean:.0f}%), uobičajen ritam",
        )

    return draw


if __name__ == "__main__":
    fc.main(
        make_model,
        "kriva zasićenja",
        "Izlaznost = konačan broj birača × kada dolaze (Weibull). "
        "Tačkasto: isti broj birača, ali uobičajenim ritmom.",
        "model_saturation.png",
        extra=extra,
        report=report,
    )
