"""Shared pieces for the turnout forecasting models.

Every model module defines one function:

    make_model(train) -> predict

  train   : list of Election (past elections the model may learn from)
  predict : predict(h, v, hours_out) -> DataFrame indexed by hours_out with columns
            lo, mid, hi (the 80% band and median of the true turnout at that hour)

With that one interface this module can backtest any model the same way
(leave-one-out over past elections), draw the same chart for every model, and
the ensemble can combine them.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import chart_style as cs

FIRST_HOUR = 9  # readings before this are ignored (no past election has them)
Z80 = 1.2816  # 80% band
BACKTEST_CUTOFFS = [9, 11, 12, 14, 16, 17, 18, 19]
FINAL = cs.FINAL_HOUR
OPEN = cs.POLLS_OPEN


@dataclass
class Election:
    name: str
    h: np.ndarray  # hours with a real reading (no interpolation), from FIRST_HOUR on
    v: np.ndarray  # turnout % at those hours (for past elections the last one is 20h)
    final: float | None  # turnout at 20h, None if not known yet


def logit(p):
    p = np.clip(np.asarray(p, float) / 100, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(z):
    return 100 / (1 + np.exp(-np.asarray(z, float)))


def load():
    """Past elections (with a 20h value) and the current one."""
    df, hours = cs.load_data()
    out = {}
    for name, row in df.iterrows():
        m = row.notna().values & (hours >= FIRST_HOUR)
        final = row.get(f"{FINAL}h")
        out[name] = Election(
            name,
            hours[m],
            row.values[m].astype(float),
            float(final) if pd.notna(final) else None,
        )
    past = [e for n, e in out.items() if n != cs.TARGET and e.final is not None]
    return past, out[cs.TARGET]


def interp_curve(e, grid):
    """Hourly curve of a past election, linear between readings, 0% at opening."""
    return np.interp(grid, np.r_[OPEN, e.h], np.r_[0.0, e.v])


# --- Backtest ---------------------------------------------------------------
def backtest(make_model, past, cutoffs=BACKTEST_CUTOFFS, verbose=True):
    """Leave-one-out: for each past election, train on the others and predict its
    20h turnout from readings up to each cutoff. Returns a DataFrame of results."""
    rows = []
    for e in past:
        train = [x for x in past if x.name != e.name]
        predict = make_model(train)
        for cut in cutoffs:
            k = e.h <= cut
            if cut >= FINAL or k.sum() == 0:
                continue
            res = predict(e.h[k], e.v[k], [FINAL]).loc[FINAL]
            rows.append(
                dict(
                    election=e.name,
                    cutoff=cut,
                    lo=res.lo,
                    mid=res.mid,
                    hi=res.hi,
                    actual=e.final,
                )
            )
    bt = pd.DataFrame(rows)
    bt["err"] = bt.mid - bt.actual
    bt["inside"] = (bt.lo <= bt.actual) & (bt.actual <= bt.hi)
    if verbose:
        print("Test na prošlim izborima (svaki predviđen iz ostalih), prognoza za 20h:")
        for cut, g in bt.groupby("cutoff"):
            print(
                f"  do {cut:>2}h: greška {g.err.abs().mean():5.2f} p.p.  "
                f"pristrasnost {g.err.mean():+5.2f}  u 80% opsegu {g.inside.sum()}/{len(g)}"
                f"  širina {(g.hi - g.lo).mean():4.1f}"
            )
    return bt


def summary_at(bt, hour):
    """Backtest stats at the cutoff closest to (but not after) `hour`."""
    usable = [c for c in sorted(bt.cutoff.unique()) if c <= hour] or [bt.cutoff.min()]
    g = bt[bt.cutoff == usable[-1]]
    return usable[-1], g.err.abs().mean(), g.err.mean(), g.inside.mean()


# --- Forecast for the current election --------------------------------------
def run_current(predict, target):
    h, v = target.h, target.v
    last = int(h[-1])
    path = predict(h, v, list(range(last, FINAL + 1)))
    as_of = pd.DataFrame(
        [predict(h[:i], v[:i], [FINAL]).loc[FINAL] for i in range(1, len(h) + 1)],
        index=h,
    )
    lo, mid, hi = path.loc[FINAL, ["lo", "mid", "hi"]]
    print(f"\n{cs.TARGET}: poslednje merenje {v[-1]:.2f}% u {last}h")
    print(f"Prognoza za {FINAL}h: {mid:.1f}%  (80% opseg {lo:.1f}–{hi:.1f}%)")
    for t, r in path.iloc[1:].iterrows():
        print(f"  {t:>2}h  {r.mid:5.1f}%  ({r.lo:.1f}–{r.hi:.1f})")
    print("Prognoza za 20h posle svakog merenja:")
    for t, r in as_of.iterrows():
        print(f"  posle {t:>2}h  {r.mid:5.1f}%  ({r.lo:.1f}–{r.hi:.1f})")
    return path, as_of


# --- Chart ------------------------------------------------------------------
def plot(path, as_of, past, model_name, subtitle, note, filename, extra=None):
    """Same two-panel chart for every model. `extra(ax)` can draw model-specific bits."""
    cs.apply()
    df, hours = cs.load_data()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 9.6), height_ratios=[3, 1.25])

    for name, row in df.drop(index=cs.TARGET).iterrows():
        if pd.notna(row.get(f"{FINAL}h")):
            ax.plot(
                *cs.readings(row, hours), color=cs.CONTEXT, lw=1.0, alpha=0.6, zorder=1
            )

    ax.fill_between(
        path.index,
        path.lo,
        path.hi,
        color=cs.ACCENT_SOFT,
        alpha=0.45,
        lw=0,
        zorder=2,
        label="prognoza, 80% opseg",
    )
    ax.plot(
        path.index,
        path.mid,
        color=cs.ACCENT,
        lw=2.4,
        ls=(0, (4, 3)),
        zorder=3,
        label=f"prognoza ({model_name})",
    )
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
    if extra:
        extra(ax)

    last = path.index[0]
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
    lo, mid, hi = path.loc[FINAL, ["lo", "mid", "hi"]]
    ax.plot(FINAL, mid, "o", ms=10, color=cs.ACCENT, mec=cs.PANEL, mew=2, zorder=5)
    ax.annotate(
        f"{mid:.0f}%",
        (FINAL, hi),
        xytext=(-6, 24),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=24,
        fontweight="bold",
        color=cs.ACCENT,
    )
    ax.annotate(
        f"prognoza u {FINAL}h · opseg {lo:.0f}–{hi:.0f}%",
        (FINAL, hi),
        xytext=(-6, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9.5,
        color=cs.INK_2,
    )
    cs.hour_axis(ax)
    ax.set_xlim(OPEN, FINAL + 0.4)
    top = max(85, hi + 14)
    ax.set_ylim(0, top)
    ax.set_yticks(range(0, int(top) + 1, 10))
    ax.set_yticklabels([f"{t}%" for t in range(0, int(top) + 1, 10)])
    ax.legend(loc="upper left")

    ax2.fill_between(
        as_of.index, as_of.lo, as_of.hi, color=cs.ACCENT_SOFT, alpha=0.45, lw=0
    )
    ax2.plot(
        as_of.index,
        as_of.mid,
        color=cs.ACCENT,
        lw=2.2,
        marker="o",
        ms=6,
        mec=cs.PANEL,
        mew=1.5,
    )
    for t, r in as_of.iterrows():
        ax2.annotate(
            f"{r.mid:.1f}",
            (t, r.mid),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=cs.INK_2,
        )
    ax2.set_title(f"Prognoza za {FINAL}h posle svakog merenja", fontsize=10.5)
    cs.hour_axis(ax2)
    ax2.set_xlim(ax.get_xlim())
    span = as_of.hi.max() - as_of.lo.min()
    ax2.set_ylim(as_of.lo.min() - 0.1 * span - 1, as_of.hi.max() + 0.15 * span + 1)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    ax2.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")

    cs.header(fig, f"Prognoza izlaznosti: {cs.TARGET}", subtitle)
    cs.footer(fig, note)
    cs.save(fig, filename)
    return fig


def main(make_model, model_name, subtitle, filename, extra=None, report=None):
    """Standard run: backtest, forecast, chart. `report(predict, target)` prints extras."""
    past, target = load()
    if len(target.h) == 0:
        raise SystemExit(f"Nema podataka za {cs.TARGET} od {FIRST_HOUR}h.")
    bt = backtest(make_model, past)
    predict = make_model(past)
    path, as_of = run_current(predict, target)
    if report:
        report(predict, target)
    cut, mae, bias, cover = summary_at(bt, int(target.h[-1]))
    note = (
        f"Test na {len(past)} prošlih izbora (do {cut}h): greška {mae:.1f} p.p., "
        f"pristrasnost {bias:+.1f}, u opsegu {cover:.0%}"
    )
    plot(
        path,
        as_of,
        past,
        model_name,
        subtitle,
        note,
        filename,
        extra=extra(predict, target) if extra else None,
    )
    plt.show()
    return bt, path, as_of
