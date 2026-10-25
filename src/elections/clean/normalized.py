import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Settings ---------------------------------------------------------------
CSV_PATH = "src/elections/data.csv"
EXCLUDE = ["Predsednički 2000.", "Parlamentarni 2000."]  # too little hourly data
POLLS_OPEN = 7
FINAL_HOUR = 20
LABEL_HOURS = [9, 11, 13, 15, 17]  # median values printed at these hours
OUTPUT = "izlaznost_udeo.png"

# --- Load data --------------------------------------------------------------
df = pd.read_csv(CSV_PATH, encoding="utf-8").set_index("Izbori")
df = df.apply(
    lambda col: pd.to_numeric(col.astype(str).str.replace("~", ""), errors="coerce")
)
df = df.drop(index=EXCLUDE, errors="ignore")
hours = np.array([int(c.rstrip("h")) for c in df.columns])
grid = np.arange(POLLS_OPEN, FINAL_HOUR + 1)

final_col = f"{FINAL_HOUR}h"
finished = [n for n in df.index[::-1] if pd.notna(df.loc[n, final_col])]  # newest first
unfinished = [n for n in df.index if pd.isna(df.loc[n, final_col])]

# --- Plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))
cmap = plt.get_cmap("viridis")
shares = {}

for i, name in enumerate(finished):
    row = df.loc[name]
    known = row.notna().values
    x = np.r_[POLLS_OPEN, hours[known]]
    y = np.r_[0.0, row.values[known]]
    share = y / y[-1] * 100  # % of the final turnout reached by each hour
    shares[name] = pd.Series(np.interp(grid, x, share), index=grid)
    ax.plot(
        x,
        share,
        marker="o",
        ms=3.5,
        lw=1.4,
        alpha=0.75,
        color=cmap(i / max(len(finished) - 1, 1)),
        label=name,
    )

median = pd.DataFrame(shares).median(axis=1)
ax.plot(grid, median, color="black", lw=3, ls="--", label="Medijana")
for h in LABEL_HOURS:
    ax.annotate(
        f"{median[h]:.0f}%",
        (h, median[h]),
        xytext=(-14, 10),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
    )

ax.set_xticks(grid)
ax.set_xticklabels([f"{h}h" for h in grid])
ax.set_xlim(POLLS_OPEN, FINAL_HOUR + 0.2)
ax.set_ylim(0, 105)
ax.grid(alpha=0.3)
ax.set_xlabel("Vreme")
ax.set_ylabel("% konačne izlaznosti")
ax.set_title("Koliki deo konačne izlaznosti je dostignut do svakog sata")
if unfinished:
    ax.text(
        0.01,
        0.97,
        f"Nije prikazano ({', '.join(unfinished)}): "
        "konačna izlaznost još nije poznata",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        color="#555",
    )
ax.legend(loc="lower right", fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=140)
plt.show()
