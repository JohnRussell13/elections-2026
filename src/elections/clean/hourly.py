import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- Settings ---------------------------------------------------------------
CSV_PATH = "src/elections/data.csv"
TARGET = "Parlamentarni 2026"  # shown first, in dark color
EXCLUDE = ["Predsednički 2000.", "Parlamentarni 2000."]  # too little hourly data
POLLS_OPEN = 7
FINAL_HOUR = 20
OUTPUT = "izlaznost_po_satu.png"

BLUE = "#5e81ac"
DARK = "#1f1f1f"

# --- Load data --------------------------------------------------------------
df = pd.read_csv(CSV_PATH, encoding="utf-8").set_index("Izbori")
df = df.apply(
    lambda col: pd.to_numeric(col.astype(str).str.replace("~", ""), errors="coerce")
)
df = df.drop(index=EXCLUDE, errors="ignore")
hours = np.array([int(c.rstrip("h")) for c in df.columns])

# Target first, then the rest from newest to oldest
order = [TARGET] + [n for n in df.index[::-1] if n != TARGET]

# --- Plot -------------------------------------------------------------------
ncols = 3
nrows = int(np.ceil(len(order) / ncols))
fig, axes = plt.subplots(
    nrows, ncols, figsize=(15, 3.25 * nrows), sharex=True, sharey=True
)
axes = np.atleast_2d(axes)

for ax, name in zip(axes.flat, order):
    row = df.loc[name]
    known = row.notna().values
    # Readings with a 0% anchor at poll opening
    hh = np.r_[POLLS_OPEN, hours[known]]
    vv = np.r_[0.0, row.values[known]]

    is_target = name == TARGET
    color = DARK if is_target else BLUE

    for i in range(1, len(hh)):
        start, end = hh[i - 1], hh[i]
        span = end - start
        rate = (vv[i] - vv[i - 1]) / span  # percentage points per hour

        # From poll opening to the first reading, with no reading in between
        lead = i == 1 and start == POLLS_OPEN and span > 1 and not is_target

        ax.bar(
            start + span / 2,
            rate,
            width=span - 0.12,
            color="white" if lead else color,
            edgecolor=color,
            hatch="///" if lead else None,
            alpha=1 if lead else (0.95 if span == 1 else 0.6),
            linewidth=1,
        )
        label = f"{rate:.1f}" + ("" if span == 1 else "/h")
        ax.text(
            start + span / 2,
            rate + 0.15,
            label,
            ha="center",
            fontsize=7.5,
            color="#333",
        )

    final = row.get(f"{FINAL_HOUR}h")
    suffix = f"   ({FINAL_HOUR}h: {final:.2f}%)" if pd.notna(final) else "   (u toku)"
    ax.set_title(
        name + suffix, fontsize=10.5, fontweight="bold" if is_target else "normal"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(POLLS_OPEN, FINAL_HOUR)

# Hide unused panels
for ax in axes.flat[len(order) :]:
    ax.set_visible(False)

ticks = range(POLLS_OPEN, FINAL_HOUR + 1)
for ax in axes[-1]:
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}h" for t in ticks], fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("p.p. po satu")
axes.flat[0].set_ylim(0, 8.5)

fig.legend(
    handles=[
        Patch(facecolor=BLUE, label="izmereno, 1 sat"),
        Patch(
            facecolor=BLUE,
            alpha=0.6,
            label="široka traka: nema merenja između, prikazan prosek po satu",
        ),
        Patch(
            facecolor="white",
            edgecolor=BLUE,
            hatch="///",
            label="od otvaranja (7h) do prvog merenja: prosek po satu",
        ),
    ],
    loc="lower center",
    ncol=3,
    fontsize=9.5,
    frameon=False,
)
fig.suptitle("Izlaznost po satu (porast izlaznosti u procentnim poenima)", fontsize=15)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(OUTPUT, dpi=140)
plt.show()
