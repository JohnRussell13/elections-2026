"""Turnout forecast with a Kalman filter.

Model (on the log-odds scale, so turnout always stays between 0 and 100%):
    reference(h)  = median log-odds turnout curve of past elections
    offset(h)     = how far the current election is above/below the reference
    offset(h+1)   = offset(h) + drift            drift ~ N(0, q)    [+ trend, if MODEL="trend"]
    reading(h)    = reference(h) + offset(h) + noise   noise ~ N(0, r)

q and r are not chosen by hand: they are fitted by maximum likelihood on past
elections, each one filtered against a reference built from the *other* elections.
The script also backtests the model on past elections before forecasting.
"""

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import chart_style as cs

MODEL = "level"  # "level": offset drifts randomly; "trend": offset can also keep moving
FIRST_HOUR = 9  # readings before this hour are ignored (no past election has them)
BAND_Z = 1.2816  # 80% band (1.645 for 90%)
BACKTEST_CUTOFFS = [11, 14, 16, 17]
BAND_PCT = round(math.erf(BAND_Z / math.sqrt(2)) * 100)

cs.apply()
df, hours = cs.load_data()
grid = np.arange(FIRST_HOUR - 1, cs.FINAL_HOUR + 1)
final_col = f"{cs.FINAL_HOUR}h"
refs = df.drop(index=cs.TARGET)
refs = refs[refs[final_col].notna()]


def logit(p):
    p = np.clip(np.asarray(p, float) / 100, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(z):
    return 100 / (1 + np.exp(-z))


def observed(row):
    """Real readings only (no interpolation), from FIRST_HOUR on."""
    m = row.notna().values
    h, v = hours[m], row.values[m]
    keep = h >= FIRST_HOUR
    return h[keep], v[keep]


# Full hourly curve of each past election (used only to build the reference)
curves = {
    n: pd.Series(logit(np.interp(grid, *cs.readings(r, hours))), index=grid)
    for n, r in refs.iterrows()
}


def reference(exclude=None):
    return pd.DataFrame({n: c for n, c in curves.items() if n != exclude}).median(
        axis=1
    )


# --- Kalman filter ----------------------------------------------------------
F = np.array([[1.0, 1.0], [0.0, 1.0]])
H = np.array([1.0, 0.0])


def run_filter(h, v, ref, q, r, qt):
    """Filter the readings. Returns log-likelihood, state, covariance, last hour."""
    Q = np.diag([q, qt])
    x = np.zeros(2)
    P = np.diag([4.0, 0.05 if qt > 0 else 0.0])  # vague prior on the offset
    t, loglik = h[0], 0.0
    for hh, vv in zip(h, v):
        while t < hh:
            x, P, t = F @ x, F @ P @ F.T + Q, t + 1
        S = H @ P @ H + r
        resid = logit(vv) - ref[hh] - H @ x
        loglik += -0.5 * (np.log(2 * np.pi * S) + resid**2 / S)
        K = P @ H / S
        x, P = x + K * resid, P - np.outer(K, H @ P)
    return loglik, x, P, t


def project(x, P, t, q, qt, until):
    """Offset mean and variance at hour `until`."""
    Q = np.diag([q, qt])
    while t < until:
        x, P, t = F @ x, F @ P @ F.T + Q, t + 1
    return x[0], P[0, 0]


def forecast(h, v, ref, params, until=cs.FINAL_HOUR):
    q, r, qt = params
    _, x, P, t = run_filter(h, v, ref, q, r, qt)
    m, var = project(x, P, t, q, qt, until)
    s = np.sqrt(var)
    return (
        expit(ref[until] + m - BAND_Z * s),
        expit(ref[until] + m),
        expit(ref[until] + m + BAND_Z * s),
    )


# --- Fit q, r (and trend noise) by maximum likelihood ------------------------
def total_loglik(q, r, qt):
    return sum(
        run_filter(*observed(row), reference(n), q, r, qt)[0]
        for n, row in refs.iterrows()
    )


q_grid = np.geomspace(1e-5, 5e-2, 20)
r_grid = np.geomspace(1e-5, 5e-2, 20)
qt_grid = [0.0] if MODEL == "level" else [1e-5, 1e-4, 1e-3]
best = max(
    (
        (total_loglik(q, r, qt), (q, r, qt))
        for qt in qt_grid
        for q in q_grid
        for r in r_grid
    ),
    key=lambda b: b[0],
)
params = best[1]
print(
    f"Model: {MODEL}   log-likelihood {best[0]:.1f}   "
    f"q={params[0]:.2e}  r={params[1]:.2e}  trend={params[2]:.0e}"
)

# --- Backtest on past elections (leave-one-out) ------------------------------
print("\nBacktest: predicting 20h from data up to the cutoff")
bias_at = {}
for cut in BACKTEST_CUTOFFS:
    errors, inside = [], 0
    for n, row in refs.iterrows():
        h, v = observed(row)
        k = h <= cut
        if k.sum() == 0:
            continue
        lo, mid, hi = forecast(h[k], v[k], reference(n), params)
        actual = row[final_col]
        errors.append(mid - actual)
        inside += lo <= actual <= hi
    e = np.array(errors)
    bias_at[cut] = e.mean()
    print(
        f"  do {cut:>2}h: prosečna greška {np.abs(e).mean():.2f} p.p., "
        f"pristrasnost {e.mean():+.2f}, u opsegu {inside}/{len(e)}"
    )

# --- Forecast for the current election ---------------------------------------
ref = reference()
h, v = observed(df.loc[cs.TARGET])
if len(h) == 0:
    raise SystemExit(f"Nema podataka za {cs.TARGET} od {FIRST_HOUR}h.")
last_h = h[-1]

as_of = pd.DataFrame(
    [forecast(h[:i], v[:i], ref, params) for i in range(1, len(h) + 1)],
    index=h[: len(h)],
    columns=["lo", "mid", "hi"],
)
future = np.arange(last_h, cs.FINAL_HOUR + 1)
path = pd.DataFrame(
    [forecast(h, v, ref, params, until=t) for t in future],
    index=future,
    columns=["lo", "mid", "hi"],
)
lo, point, hi = path.loc[cs.FINAL_HOUR]

print(f"\n{cs.TARGET}: poslednje merenje {v[-1]:.2f}% u {last_h}h")
print(
    f"Prognoza za {cs.FINAL_HOUR}h: {point:.1f}%  "
    f"({BAND_PCT}% opseg {lo:.1f}–{hi:.1f}%)"
)
for t, row in path.iloc[1:].iterrows():
    print(f"  {t:>2}h  {row.mid:5.1f}%  ({row.lo:.1f}–{row.hi:.1f})")
print("\nPrognoza za 20h kroz dan:")
for t, row in as_of.iterrows():
    print(f"  posle {t:>2}h  {row.mid:5.1f}%  ({row.lo:.1f}–{row.hi:.1f})")

# --- Plot -------------------------------------------------------------------
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 9.6), height_ratios=[3, 1.25])

for _, row in refs.iterrows():
    ax.plot(*cs.readings(row, hours), color=cs.CONTEXT, lw=1.0, alpha=0.6, zorder=1)

ax.fill_between(
    path.index,
    path.lo,
    path.hi,
    color=cs.ACCENT_SOFT,
    alpha=0.45,
    lw=0,
    zorder=2,
    label=f"prognoza, {BAND_PCT}% opseg",
)
ax.plot(
    path.index,
    path.mid,
    color=cs.ACCENT,
    lw=2.4,
    ls=(0, (4, 3)),
    zorder=3,
    label="prognoza (Kalmanov filter)",
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
ax.annotate(
    f"{v[-1]:.1f}%",
    (last_h, v[-1]),
    xytext=(10, -16),
    textcoords="offset points",
    ha="left",
    fontsize=11,
    fontweight="bold",
    color=cs.INK,
)
ax.plot(
    cs.FINAL_HOUR, point, "o", ms=10, color=cs.ACCENT, mec=cs.PANEL, mew=2, zorder=5
)
ax.annotate(
    f"{point:.0f}%",
    (cs.FINAL_HOUR, hi),
    xytext=(-6, 24),
    textcoords="offset points",
    ha="right",
    va="bottom",
    fontsize=24,
    fontweight="bold",
    color=cs.ACCENT,
)
ax.annotate(
    f"prognoza u {cs.FINAL_HOUR}h · opseg {lo:.0f}–{hi:.0f}%",
    (cs.FINAL_HOUR, hi),
    xytext=(-6, 8),
    textcoords="offset points",
    ha="right",
    va="bottom",
    fontsize=9.5,
    color=cs.INK_2,
)
cs.hour_axis(ax)
ax.set_xlim(cs.POLLS_OPEN, cs.FINAL_HOUR + 0.4)
ax.set_ylim(0, max(85, hi + 14))
ax.set_yticks(range(0, int(ax.get_ylim()[1]) + 1, 10))
ax.set_yticklabels([f"{t}%" for t in ax.get_yticks()])
ax.legend(loc="upper left")

# How the 20h forecast changed as readings came in
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
for t, row in as_of.iterrows():
    ax2.annotate(
        f"{row.mid:.1f}",
        (t, row.mid),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=8.5,
        color=cs.INK_2,
    )
ax2.set_title(f"Prognoza za {cs.FINAL_HOUR}h posle svakog merenja", fontsize=10.5)
cs.hour_axis(ax2)
ax2.set_xlim(ax.get_xlim())
ax2.set_ylim(as_of.lo.min() - 2, as_of.hi.max() + 3)
ax2.yaxis.set_major_locator(plt.MultipleLocator(2))
ax2.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")

cs.header(
    fig,
    f"Prognoza izlaznosti: {cs.TARGET}",
    f"Kalmanov filter na log-odds skali; parametri procenjeni na "
    f"{len(refs)} prošlih izbora. Merenja do {last_h}h.",
)
cut_note = max(c for c in BACKTEST_CUTOFFS if c <= max(last_h, min(BACKTEST_CUTOFFS)))
cs.footer(
    fig,
    f"Model: {MODEL}; opseg = {BAND_PCT}%; test na prošlim izborima (do {cut_note}h): "
    f"pristrasnost {bias_at[cut_note]:+.1f} p.p.",
)
cs.save(fig, "kalman.png")
plt.show()
