"""Model 6: ensemble of all models, weighted by how well each did on past elections.

  1. Every model is backtested the same way (leave-one-out over past elections).
  2. Weight of a model = 1 / its mean squared error at the backtest cutoff matching
     the latest reading today (models that were better at this time of day count more).
  3. The forecast is a mixture of the models' forecasts with those weights, so the
     band also widens when the models disagree.

The ensemble is itself backtested: for each past election the weights are computed
from the *other* elections only, so the test is honest.
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import chart_style as cs
import forecast_common as fc
import model_bayes
import model_gp
import model_kalman
import model_regression
import model_saturation

MODELS = {
    "Kalmanov filter": model_kalman,
    "kriva zasićenja": model_saturation,
    "Gausov proces": model_gp,
    "regresija": model_regression,
    "hijerarhijski Bayes": model_bayes,
}
N_SAMPLES = 40000
SEED = 0


def to_normal(df):
    """Approximate a model's forecast as a normal on the log-odds scale."""
    mu = fc.logit(df.mid.values)
    sd = (fc.logit(df.hi.values) - fc.logit(df.lo.values)) / (2 * fc.Z80)
    return mu, np.clip(sd, 1e-4, None)


def mixture(forecasts, weights):
    """forecasts: {model: DataFrame lo/mid/hi}; returns combined DataFrame."""
    rng = np.random.default_rng(SEED)
    hours = next(iter(forecasts.values())).index
    names = list(forecasts)
    w = np.array([weights[n] for n in names])
    counts = rng.multinomial(N_SAMPLES, w / w.sum())
    draws = []
    for n, c in zip(names, counts):
        mu, sd = to_normal(forecasts[n])
        draws.append(mu[None, :] + sd[None, :] * rng.standard_normal((c, len(hours))))
    q = np.quantile(fc.expit(np.vstack(draws)), [0.1, 0.5, 0.9], axis=0)
    return pd.DataFrame(q.T, index=hours, columns=["lo", "mid", "hi"])


def weights_from(backtests, cutoff, exclude=None):
    w = {}
    for name, bt in backtests.items():
        g = bt[(bt.cutoff == cutoff) & (bt.election != exclude)]
        w[name] = 1 / max((g.err**2).mean(), 1e-6)
    total = sum(w.values())
    return {n: x / total for n, x in w.items()}


def ensemble_backtest(backtests):
    """Combine the models' leave-one-out forecasts with leave-one-out weights."""
    any_bt = next(iter(backtests.values()))
    rows = []
    for (election, cutoff), _ in any_bt.groupby(["election", "cutoff"]):
        w = weights_from(backtests, cutoff, exclude=election)
        fcs = {}
        for name, bt in backtests.items():
            r = bt[(bt.election == election) & (bt.cutoff == cutoff)].iloc[0]
            fcs[name] = pd.DataFrame(
                [[r.lo, r.mid, r.hi]], index=[fc.FINAL], columns=["lo", "mid", "hi"]
            )
            actual = r.actual
        m = mixture(fcs, w).loc[fc.FINAL]
        rows.append(
            dict(
                election=election,
                cutoff=cutoff,
                lo=m.lo,
                mid=m.mid,
                hi=m.hi,
                actual=actual,
            )
        )
    bt = pd.DataFrame(rows)
    bt["err"] = bt.mid - bt.actual
    bt["inside"] = (bt.lo <= bt.actual) & (bt.actual <= bt.hi)
    return bt


def print_table(bt, label):
    print(f"\n{label}")
    for cut, g in bt.groupby("cutoff"):
        print(
            f"  do {cut:>2}h: greška {g.err.abs().mean():5.2f} p.p.  "
            f"pristrasnost {g.err.mean():+5.2f}  u 80% opsegu {g.inside.sum()}/{len(g)}"
            f"  širina {(g.hi - g.lo).mean():4.1f}"
        )


def main():
    past, target = fc.load()
    last = int(target.h[-1])
    backtests, predictors = {}, {}
    for name, module in MODELS.items():
        print(f"Testiram: {name} ...", flush=True)
        backtests[name] = fc.backtest(module.make_model, past, verbose=False)
        predictors[name] = module.make_model(past)

    cutoff = fc.summary_at(next(iter(backtests.values())), last)[0]
    print(f"\nTačnost modela na prošlim izborima (do {cutoff}h) i težine:")
    weights = weights_from(backtests, cutoff)
    for name in MODELS:
        g = backtests[name][backtests[name].cutoff == cutoff]
        print(
            f"  {name:<20} greška {g.err.abs().mean():4.2f} p.p.  "
            f"u opsegu {g.inside.mean():4.0%}  težina {weights[name]:5.1%}"
        )

    ens_bt = ensemble_backtest(backtests)
    print_table(ens_bt, "Ansambl na prošlim izborima (težine bez tog izbora):")

    hours_out = list(range(last, fc.FINAL + 1))
    paths = {n: p(target.h, target.v, hours_out) for n, p in predictors.items()}
    path = mixture(paths, weights)

    as_of_rows = []
    for i in range(1, len(target.h) + 1):
        h, v = target.h[:i], target.v[:i]
        cut_i = fc.summary_at(ens_bt, int(h[-1]))[0]
        w_i = weights_from(backtests, cut_i)
        f_i = {n: p(h, v, [fc.FINAL]) for n, p in predictors.items()}
        as_of_rows.append(mixture(f_i, w_i).loc[fc.FINAL])
    as_of = pd.DataFrame(as_of_rows, index=target.h)

    lo, mid, hi = path.loc[fc.FINAL, ["lo", "mid", "hi"]]
    print(f"\n{cs.TARGET}: poslednje merenje {target.v[-1]:.2f}% u {last}h")
    print(
        f"ANSAMBL, prognoza za {fc.FINAL}h: {mid:.1f}%  (80% opseg {lo:.1f}–{hi:.1f}%)"
    )
    for name, p in paths.items():
        r = p.loc[fc.FINAL]
        print(
            f"  {name:<20} {r.mid:5.1f}%  ({r.lo:.1f}–{r.hi:.1f})  težina {weights[name]:.0%}"
        )
    for t, r in path.iloc[1:].iterrows():
        print(f"  {t:>2}h  {r.mid:5.1f}%  ({r.lo:.1f}–{r.hi:.1f})")

    plot(path, paths, weights, as_of, ens_bt, cutoff, last)
    plt.show()


def plot(path, paths, weights, as_of, ens_bt, cutoff, last):
    cs.apply()
    df, hours = cs.load_data()
    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[3, 1.35], width_ratios=[1.25, 1], hspace=0.32, wspace=0.18
    )
    ax = fig.add_subplot(gs[0, :])
    ax_models = fig.add_subplot(gs[1, 0])
    ax_time = fig.add_subplot(gs[1, 1])

    for name, row in df.drop(index=cs.TARGET).iterrows():
        if pd.notna(row.get(f"{fc.FINAL}h")):
            ax.plot(
                *cs.readings(row, hours), color=cs.CONTEXT, lw=1.0, alpha=0.5, zorder=1
            )
    for name, p in paths.items():
        ax.plot(p.index, p.mid, color=cs.PAST, lw=1.0, alpha=0.55, zorder=2)
    ax.fill_between(
        path.index,
        path.lo,
        path.hi,
        color=cs.ACCENT_SOFT,
        alpha=0.45,
        lw=0,
        zorder=2,
        label="ansambl, 80% opseg",
    )
    ax.plot(
        path.index,
        path.mid,
        color=cs.ACCENT,
        lw=2.6,
        ls=(0, (4, 3)),
        zorder=3,
        label="ansambl",
    )
    ax.plot([], [], color=cs.PAST, lw=1.0, alpha=0.55, label="pojedinačni modeli")
    all_h, all_v = cs.readings(df.loc[cs.TARGET], hours)
    ax.plot(
        all_h[1:],
        all_v[1:],
        color=cs.ACCENT,
        lw=3,
        marker="o",
        ms=8,
        mec=cs.PANEL,
        mew=2,
        zorder=4,
        label=f"{cs.TARGET}, izmereno",
    )
    ax.annotate(
        f"{all_v[-1]:.1f}%",
        (all_h[-1], all_v[-1]),
        xytext=(10, -16),
        textcoords="offset points",
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=cs.INK,
    )
    lo, mid, hi = path.loc[fc.FINAL, ["lo", "mid", "hi"]]
    ax.plot(fc.FINAL, mid, "o", ms=10, color=cs.ACCENT, mec=cs.PANEL, mew=2, zorder=5)
    ax.annotate(
        f"{mid:.0f}%",
        (fc.FINAL, hi),
        xytext=(-6, 24),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=24,
        fontweight="bold",
        color=cs.ACCENT,
    )
    ax.annotate(
        f"ansambl u {fc.FINAL}h · opseg {lo:.0f}–{hi:.0f}%",
        (fc.FINAL, hi),
        xytext=(-6, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=cs.INK_2,
    )
    cs.hour_axis(ax)
    ax.set_xlim(fc.OPEN, fc.FINAL + 0.4)
    top = max(85, hi + 14)
    ax.set_ylim(0, top)
    ax.set_yticks(range(0, int(top) + 1, 10))
    ax.set_yticklabels([f"{t}%" for t in range(0, int(top) + 1, 10)])
    ax.legend(loc="upper left")

    # Each model's 20h forecast with its band and weight
    names = list(paths)
    ys = np.arange(len(names))[::-1]
    for y, n in zip(ys, names):
        r = paths[n].loc[fc.FINAL]
        ax_models.plot(
            [r.lo, r.hi], [y, y], color=cs.PAST_SOFT, lw=5, solid_capstyle="round"
        )
        ax_models.plot(r.mid, y, "o", ms=8, color=cs.PAST, mec=cs.PANEL, mew=1.5)
        ax_models.text(
            r.hi + 0.4,
            y,
            f"{r.mid:.1f}%  ·  težina {weights[n]:.0%}",
            va="center",
            fontsize=8.5,
            color=cs.INK_2,
        )
    ax_models.axvspan(lo, hi, color=cs.ACCENT_SOFT, alpha=0.35, lw=0)
    ax_models.axvline(mid, color=cs.ACCENT, lw=2)
    ax_models.set_yticks(ys)
    ax_models.set_yticklabels(names)
    ax_models.grid(axis="x")
    ax_models.grid(axis="y", visible=False)
    xmin = min(paths[n].loc[fc.FINAL].lo for n in names)
    xmax = max(paths[n].loc[fc.FINAL].hi for n in names)
    ax_models.set_xlim(xmin - 1, xmax + 9)
    ax_models.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax_models.set_title(
        f"Modeli: prognoza za {fc.FINAL}h (crveno = ansambl)", fontsize=10.5
    )

    ax_time.fill_between(
        as_of.index, as_of.lo, as_of.hi, color=cs.ACCENT_SOFT, alpha=0.45, lw=0
    )
    ax_time.plot(
        as_of.index,
        as_of.mid,
        color=cs.ACCENT,
        lw=2.2,
        marker="o",
        ms=5,
        mec=cs.PANEL,
        mew=1.2,
    )
    for t, r in as_of.iterrows():
        ax_time.annotate(
            f"{r.mid:.0f}",
            (t, r.mid),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=cs.INK_2,
        )
    ax_time.set_xticks(as_of.index)
    ax_time.set_xticklabels([f"{t}h" for t in as_of.index], fontsize=8)
    ax_time.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    ax_time.set_title(f"Ansambl: prognoza za {fc.FINAL}h kroz dan", fontsize=10.5)

    g = ens_bt[ens_bt.cutoff == cutoff]
    cs.header(
        fig,
        f"Prognoza izlaznosti: {cs.TARGET}",
        f"Ansambl {len(paths)} modela, težine prema tačnosti na prošlim izborima "
        f"u ovo doba dana. Merenja do {last}h.",
    )
    cs.footer(
        fig,
        f"Ansambl na prošlim izborima (do {cutoff}h): greška "
        f"{g.err.abs().mean():.1f} p.p., u opsegu {g.inside.mean():.0%}",
    )
    cs.save(fig, "model_ensemble.png")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*tight_layout.*")
    main()
