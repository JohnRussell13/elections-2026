import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Settings ---------------------------------------------------------------
CSV_PATH = "src/elections/data.csv"
TARGET = "Parlamentarni 2026"
# Elections left out of the reference profile (too sparse / approximate / atypical)
EXCLUDE = [
    "Predsednički 2000.",
    "Parlamentarni 2000.",
]  # e.g. add "Parlamentarni 2020."
# Use only the K past elections whose curves best match the target so far (None = all)
NEAREST_K = None
# "logit": shift past curves on the log-odds scale (bounded below 100%, best when
#          2026 runs far from past elections). "ratio": scale past curves proportionally.
METHOD = "logit"
POLLS_OPEN = 7  # turnout is 0% when polls open
FINAL_HOUR = 20

# --- Load data --------------------------------------------------------------
df = pd.read_csv(CSV_PATH, encoding="utf-8").set_index("Izbori")
df = df.apply(
    lambda col: pd.to_numeric(col.astype(str).str.replace("~", ""), errors="coerce")
)
hours = np.array([int(h.rstrip("h")) for h in df.columns])
grid = np.arange(POLLS_OPEN, FINAL_HOUR + 1)  # 7h..20h, hourly


def full_curve(row):
    """Interpolate an election's curve over 7h..20h, anchored at 0% at 7h."""
    known = row.notna().values
    x = np.concatenate([[POLLS_OPEN], hours[known]])
    y = np.concatenate([[0.0], row.values[known]])
    return np.interp(grid, x, y)


# --- Reference profiles: share of final turnout reached at each hour --------
refs = df.drop(index=[TARGET, *EXCLUDE], errors="ignore")
refs = refs[refs[f"{FINAL_HOUR}h"].notna()]
curves = pd.DataFrame(
    {name: full_curve(row) for name, row in refs.iterrows()}, index=grid
).T
profiles = curves.div(curves[FINAL_HOUR], axis=0)  # rows: elections, cols: hours

# --- Target observations ----------------------------------------------------
target = df.loc[TARGET]
obs_mask = target.notna().values
obs_h = hours[obs_mask]
obs_v = target.values[obs_mask]
if len(obs_h) == 0:
    raise SystemExit(f"No data yet for {TARGET}.")
first_ref_hour = hours[refs.notna().any().values].min()
if obs_h.min() < first_ref_hour:
    print(
        f"WARNING: {TARGET} has readings before {first_ref_hour}h, where no past "
        f"election has data. Those hours are only estimated (linear from 0% at "
        f"{POLLS_OPEN}h), so the forecast may be badly off.\n"
    )


def logit(p):
    p = np.clip(p / 100, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(z):
    return 100 / (1 + np.exp(-z))


if METHOD == "ratio":
    # Scale each past profile so it fits the observed 2026 points (least squares).
    P = profiles[obs_h].values  # shape (n_refs, n_obs)
    scale = (P @ obs_v) / (P**2).sum(axis=1)
    projections = profiles.mul(scale, axis=0)
else:
    # Shift each past curve on the log-odds scale. Turnout can never exceed 100%,
    # and a big early lead translates into a smaller lead later in the day.
    L = logit(curves[obs_h].values)
    shift = (logit(obs_v)[None, :] - L).mean(axis=1)
    projections = pd.DataFrame(
        expit(logit(curves.values) + shift[:, None]),
        index=curves.index,
        columns=curves.columns,
    )

finals = projections[FINAL_HOUR]

# Optionally keep only the most similar past elections
if NEAREST_K:
    err = ((projections[obs_h].values - obs_v) ** 2).mean(axis=1)
    keep = pd.Series(err, index=projections.index).nsmallest(NEAREST_K).index
    finals, projections = finals[keep], projections.loc[keep]

# --- Forecast ---------------------------------------------------------------
point = finals.median()
lo, hi = finals.quantile(0.1), finals.quantile(0.9)

proj_mid = projections.median()
proj_lo, proj_hi = projections.quantile(0.1), projections.quantile(0.9)

last_h = obs_h.max()
print(f"{TARGET}: latest reading {obs_v[-1]:.2f}% at {last_h}h")
print(f"Forecast final turnout (20h): {point:.1f}%  (80% range {lo:.1f}–{hi:.1f}%)")
print("\nImplied by each reference election:")
for name, val in finals.sort_values().items():
    print(f"  {name:<28} {val:5.1f}%")
print("\nProjected curve:")
for h in grid[grid > last_h]:
    print(f"  {h:>2}h  {proj_mid[h]:5.1f}%  ({proj_lo[h]:.1f}–{proj_hi[h]:.1f})")

# --- Plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))
cmap = plt.get_cmap("tab20")

for i, (name, row) in enumerate(df.drop(index=TARGET).iterrows()):
    mask = row.notna().values
    ax.plot(
        hours[mask],
        row.values[mask],
        marker="o",
        linewidth=1.2,
        markersize=3,
        alpha=0.45,
        label=name,
        color=cmap(i),
    )

future = grid[grid >= last_h]
ax.fill_between(
    future,
    proj_lo[future],
    proj_hi[future],
    color="black",
    alpha=0.12,
    label="Prognoza (80% opseg)",
)
ax.plot(
    future,
    proj_mid[future],
    color="black",
    linestyle="--",
    linewidth=2,
    label="Prognoza (medijana)",
)
ax.plot(
    obs_h, obs_v, color="black", marker="o", linewidth=3, markersize=7, label=TARGET
)
ax.annotate(
    f"{point:.1f}%\n({lo:.1f}–{hi:.1f})",
    (FINAL_HOUR, point),
    xytext=(8, 0),
    textcoords="offset points",
    va="center",
    fontsize=10,
    fontweight="bold",
)

ax.set_xticks(hours)
ax.set_xticklabels([f"{h}h" for h in hours])
ax.set_xlabel("Vreme")
ax.set_ylabel("Izlaznost (%)")
ax.set_title(f"Izlaznost po satima i prognoza za {TARGET}")
ax.set_ylim(0, max(80, hi + 5))
ax.grid(True, alpha=0.3)
ax.legend(loc="center left", bbox_to_anchor=(1.08, 0.5), fontsize=9)
plt.tight_layout()
plt.savefig("izlaznost_prognoza.png", dpi=150, bbox_inches="tight")
plt.show()
