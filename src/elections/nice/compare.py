"""Compare two forecasts of the final turnout for the current election:
one made from the first readings only, and one from all readings so far.

Uses the same "logit" method as model2.py: each past election's curve is shifted on
the log-odds scale to fit the current readings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import chart_style as cs

EARLY_POINTS = 2  # how many of the first readings the early forecast uses

cs.apply()
df, hours = cs.load_data()
grid = np.arange(cs.POLLS_OPEN, cs.FINAL_HOUR + 1)
final_col = f"{cs.FINAL_HOUR}h"

# --- Reference curves -------------------------------------------------------
refs = df.drop(index=cs.TARGET)
refs = refs[refs[final_col].notna()]
curves = pd.DataFrame(
    {n: np.interp(grid, *cs.readings(r, hours)) for n, r in refs.iterrows()}, index=grid
).T

obs_h, obs_v = cs.readings(df.loc[cs.TARGET], hours)
obs_h, obs_v = obs_h[1:], obs_v[1:]  # drop the 0% anchor
if len(obs_h) <= EARLY_POINTS:
    raise SystemExit(f"Potrebno je više od {EARLY_POINTS} merenja za poređenje.")


def logit(p):
    p = np.clip(np.asarray(p) / 100, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(z):
    return 100 / (1 + np.exp(-z))


def forecast(h, v):
    """Median curve, 10th/90th percentile band and final numbers from readings h, v."""
    shift = (logit(v)[None, :] - logit(curves[h].values)).mean(axis=1)
    proj = pd.DataFrame(
        expit(logit(curves.values) + shift[:, None]),
        index=curves.index,
        columns=curves.columns,
    )
    finals = proj[cs.FINAL_HOUR]
    return {
        "last_h": h.max(),
        "mid": proj.median(),
        "lo": proj.quantile(0.1),
        "hi": proj.quantile(0.9),
        "point": finals.median(),
        "f_lo": finals.quantile(0.1),
        "f_hi": finals.quantile(0.9),
    }


early = forecast(obs_h[:EARLY_POINTS], obs_v[:EARLY_POINTS])
latest = forecast(obs_h, obs_v)

for label, f in [("Prva merenja", early), ("Najnovija merenja", latest)]:
    print(
        f"{label} (do {f['last_h']}h): {f['point']:.1f}%  "
        f"(80% opseg {f['f_lo']:.1f}–{f['f_hi']:.1f}%)"
    )

# --- Plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 7.2))

for _, row in refs.iterrows():
    ax.plot(*cs.readings(row, hours), color=cs.CONTEXT, lw=1.0, alpha=0.7, zorder=1)

styles = [
    (
        early,
        cs.PAST,
        cs.PAST_SOFT,
        f"prognoza iz prva {EARLY_POINTS} merenja (do {early['last_h']}h)",
    ),
    (
        latest,
        cs.ACCENT,
        cs.ACCENT_SOFT,
        f"prognoza iz svih merenja (do {latest['last_h']}h)",
    ),
]
for f, color, soft, label in styles:
    future = grid[grid >= f["last_h"]]
    ax.fill_between(
        future, f["lo"][future], f["hi"][future], color=soft, alpha=0.35, lw=0, zorder=2
    )
    ax.plot(
        future,
        f["mid"][future],
        color=color,
        lw=2.4,
        ls=(0, (4, 3)),
        zorder=3,
        label=label,
    )

# Measured readings; the ones used by the early forecast are drawn in black
ax.plot(obs_h, obs_v, color=cs.ACCENT, lw=3, zorder=4)
ax.plot(
    obs_h[EARLY_POINTS:],
    obs_v[EARLY_POINTS:],
    "o",
    ms=8,
    color=cs.ACCENT,
    mec=cs.PANEL,
    mew=2,
    zorder=5,
    label=f"{cs.TARGET}, izmereno",
)
ax.plot(
    obs_h[:EARLY_POINTS],
    obs_v[:EARLY_POINTS],
    "o",
    ms=8,
    color=cs.PAST,
    mec=cs.PANEL,
    mew=2,
    zorder=5,
    label="prva merenja",
)
ax.annotate(
    f"{obs_v[-1]:.1f}%",
    (obs_h[-1], obs_v[-1]),
    xytext=(-8, 10),
    textcoords="offset points",
    ha="right",
    fontsize=11,
    fontweight="bold",
    color=cs.INK,
)

# Final numbers at 20h, one above the other so they never overlap
ordered = sorted(
    [(early, cs.PAST, "prva merenja"), (latest, cs.ACCENT, "sva merenja")],
    key=lambda t: t[0]["point"],
    reverse=True,
)
top = max(early["f_hi"], latest["f_hi"])
for i, (f, color, name) in enumerate(ordered):
    ax.plot(
        cs.FINAL_HOUR,
        f["point"],
        "o",
        ms=10,
        color=color,
        mec=cs.PANEL,
        mew=2,
        zorder=6,
    )
    y_off = 70 - i * 52
    ax.annotate(
        f"{f['point']:.1f}%",
        (cs.FINAL_HOUR, top),
        xytext=(-6, y_off),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=22,
        fontweight="bold",
        color=color,
    )
    ax.annotate(
        f"{name} · opseg {f['f_lo']:.0f}–{f['f_hi']:.0f}%",
        (cs.FINAL_HOUR, top),
        xytext=(-6, y_off - 14),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        color=cs.INK_2,
    )

cs.hour_axis(ax)
ax.set_xlim(cs.POLLS_OPEN, cs.FINAL_HOUR + 0.6)
ax.set_ylim(0, max(85, top + 22))
ax.set_yticks(range(0, int(ax.get_ylim()[1]) + 1, 10))
ax.set_yticklabels([f"{t}%" for t in ax.get_yticks()])
ax.legend(loc="upper left")

change = latest["point"] - early["point"]
cs.header(
    fig,
    f"Prognoza izlaznosti: prva vs. najnovija merenja",
    f"Prognoza za {cs.FINAL_HOUR}h iz merenja do {early['last_h']}h "
    f"({early['point']:.1f}%) i do {latest['last_h']}h ({latest['point']:.1f}%): "
    f"razlika {change:+.1f} p.p.",
)
cs.footer(
    fig,
    "Metod: log-odds pomeranje krivih prošlih izbora (sivo); "
    "senka = 10.–90. percentil",
)
cs.save(fig, "model_compare.png")
plt.show()
