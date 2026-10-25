"""Forecast of the final turnout for the current election from its hourly readings.

Method "logit": each past election's curve is shifted on the log-odds scale to fit
the current readings, so the forecast can never exceed 100% and an early lead
shrinks later in the day. "ratio": past curves are scaled proportionally.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import chart_style as cs

METHOD = "logit"
NEAREST_K = None  # use only the K past elections that match best so far (None = all)

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
profiles = curves.div(curves[cs.FINAL_HOUR], axis=0)

obs_h, obs_v = cs.readings(df.loc[cs.TARGET], hours)
obs_h, obs_v = obs_h[1:], obs_v[1:]  # drop the 0% anchor
if len(obs_h) == 0:
    raise SystemExit(f"Nema podataka za {cs.TARGET}.")


def logit(p):
    p = np.clip(np.asarray(p) / 100, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(z):
    return 100 / (1 + np.exp(-z))


if METHOD == "ratio":
    P = profiles[obs_h].values
    scale = (P @ obs_v) / (P**2).sum(axis=1)
    projections = profiles.mul(scale, axis=0)
else:
    shift = (logit(obs_v)[None, :] - logit(curves[obs_h].values)).mean(axis=1)
    projections = pd.DataFrame(
        expit(logit(curves.values) + shift[:, None]),
        index=curves.index,
        columns=curves.columns,
    )

if NEAREST_K:
    err = ((projections[obs_h].values - obs_v) ** 2).mean(axis=1)
    keep = pd.Series(err, index=projections.index).nsmallest(NEAREST_K).index
    projections = projections.loc[keep]

finals = projections[cs.FINAL_HOUR]
point, lo, hi = finals.median(), finals.quantile(0.1), finals.quantile(0.9)
mid = projections.median()
band_lo, band_hi = projections.quantile(0.1), projections.quantile(0.9)
last_h = obs_h.max()

print(f"{cs.TARGET}: poslednje merenje {obs_v[-1]:.2f}% u {last_h}h")
print(f"Prognoza za {cs.FINAL_HOUR}h: {point:.1f}%  (80% opseg {lo:.1f}–{hi:.1f}%)")
for h in grid[grid > last_h]:
    print(f"  {h:>2}h  {mid[h]:5.1f}%  ({band_lo[h]:.1f}–{band_hi[h]:.1f})")

# --- Plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 7.2))

end_labels = []
for name, row in refs.iterrows():
    x, y = cs.readings(row, hours)
    ax.plot(x, y, color=cs.CONTEXT, lw=1.1, zorder=1)
    end_labels.append((y[-1], cs.short(name)))

# Past-election names at the right edge, nudged apart so they never overlap
MIN_GAP = 1.9
end_labels.sort()
placed = []
for y, label in end_labels:
    y_pos = max(y, placed[-1] + MIN_GAP) if placed else y
    placed.append(y_pos)
    ax.text(
        cs.FINAL_HOUR + 0.12, y_pos, label, fontsize=7.5, color=cs.INK_3, va="center"
    )

future = grid[grid >= last_h]
ax.fill_between(
    future,
    band_lo[future],
    band_hi[future],
    color=cs.ACCENT_SOFT,
    alpha=0.45,
    lw=0,
    zorder=2,
    label="prognoza, 80% opseg",
)
ax.plot(
    future,
    mid[future],
    color=cs.ACCENT,
    lw=2.2,
    ls=(0, (4, 3)),
    zorder=3,
    label="prognoza, medijana",
)
ax.plot(
    obs_h,
    obs_v,
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
    f"{obs_v[-1]:.1f}%",
    (last_h, obs_v[-1]),
    xytext=(-8, 10),
    textcoords="offset points",
    ha="right",
    fontsize=11,
    fontweight="bold",
    color=cs.INK,
)

# Headline number at 20h
ax.plot(
    cs.FINAL_HOUR, point, "o", ms=10, color=cs.ACCENT, mec=cs.PANEL, mew=2, zorder=5
)
# Headline sits above the top of the forecast band so it never covers the line
ax.annotate(
    f"{point:.0f}%",
    (cs.FINAL_HOUR, hi),
    xytext=(-6, 26),
    textcoords="offset points",
    ha="right",
    va="bottom",
    fontsize=24,
    fontweight="bold",
    color=cs.INK,
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
ax.set_xlim(cs.POLLS_OPEN, cs.FINAL_HOUR + 1.1)
ax.set_ylim(0, max(85, hi + 14))
ax.set_yticks(range(0, int(ax.get_ylim()[1]) + 1, 10))
ax.set_yticklabels([f"{t}%" for t in ax.get_yticks()])
ax.legend(loc="upper left")

method = "log-odds pomeranje" if METHOD == "logit" else "proporcionalno skaliranje"
cs.header(
    fig,
    f"Prognoza izlaznosti: {cs.TARGET}",
    f"Očekivana izlaznost u {cs.FINAL_HOUR}h na osnovu merenja do {last_h}h, "
    "u poređenju sa tokom dana na prošlim izborima (sivo).",
)
cs.footer(
    fig,
    f"Metod: {method} krivih {len(projections)} prošlih izbora; "
    "opseg = 10.–90. percentil",
)
cs.save(fig, "model2.png")
plt.show()
