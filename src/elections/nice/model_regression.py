"""Model 3: hourly regression ("nowcasting").

For the latest hour L with a reading, fit across past elections (log-odds scale):

    logit turnout(t) = a + b · logit turnout(L) + c · momentum

  momentum = logit turnout(L) − logit turnout(L − MOMENTUM_HOURS)
             (has the pace been speeding up or slowing down lately?)

one regression per future hour t (17h, 18h, ..., 20h). Prediction intervals are
the usual OLS ones (Student-t), so they widen automatically when today is far
from the past elections (as 2026 is).

Past elections do not have a reading every hour, so their value at L is taken
from the straight line between their nearest readings.
"""

import numpy as np
import pandas as pd

import forecast_common as fc

MOMENTUM_HOURS = 2  # 0 = no momentum term
T90 = {
    2: 1.886,
    3: 1.638,
    4: 1.533,
    5: 1.476,
    6: 1.440,
    7: 1.415,
    8: 1.397,
    9: 1.383,
    10: 1.372,
    11: 1.363,
    12: 1.356,
}  # Student-t 90th percentile

GRID = np.arange(fc.OPEN, fc.FINAL + 1)


def make_model(train):
    curves = pd.DataFrame(
        [fc.interp_curve(e, GRID) for e in train],
        index=[e.name for e in train],
        columns=GRID,
    )
    L = fc.logit(curves.values)
    Lc = pd.DataFrame(L, index=curves.index, columns=GRID)

    def features(level_now, level_before, use_mom):
        cols = [np.ones_like(level_now), level_now]
        if use_mom:
            cols.append(level_now - level_before)
        return np.column_stack(cols)

    def predict(h, v, hours_out):
        last = int(h[-1])
        y_now = fc.logit(v[-1])
        before = last - MOMENTUM_HOURS
        use_mom = MOMENTUM_HOURS > 0 and before in h and before >= fc.FIRST_HOUR
        y_before = fc.logit(v[list(h).index(before)]) if use_mom else 0.0
        X = features(Lc[last].values, Lc[before].values if use_mom else None, use_mom)
        x0 = features(np.array([y_now]), np.array([y_before]), use_mom)[0]
        XtX_inv = np.linalg.inv(X.T @ X)
        n, p = X.shape
        t = T90[min(max(n - p, 2), 12)]
        rows = []
        for hour in hours_out:
            if hour <= last:
                y = fc.logit(np.interp(hour, h, v)) if hour >= h[0] else y_now
                rows.append((fc.expit(y),) * 3)
                continue
            target = Lc[hour].values
            beta = XtX_inv @ X.T @ target
            resid = target - X @ beta
            s2 = resid @ resid / max(n - p, 1)
            mu = x0 @ beta
            se = np.sqrt(s2 * (1 + x0 @ XtX_inv @ x0))
            rows.append((fc.expit(mu - t * se), fc.expit(mu), fc.expit(mu + t * se)))
        return pd.DataFrame(rows, index=hours_out, columns=["lo", "mid", "hi"])

    predict.curves = curves
    return predict


def report(predict, target):
    last = int(target.h[-1])
    past_now = predict.curves[last]
    print(
        f"\nU {last}h: danas {target.v[-1]:.1f}%, prošli izbori "
        f"{past_now.min():.1f}–{past_now.max():.1f}% "
        f"→ regresija ekstrapolira van poznatog opsega"
        if target.v[-1] > past_now.max()
        else ""
    )


if __name__ == "__main__":
    fc.main(
        make_model,
        "regresija po satu",
        f"Regresija: izlaznost u 20h prema izlaznosti u poslednjem satu "
        f"i promeni u poslednja {MOMENTUM_HOURS} sata (prošli izbori).",
        "model_regression.png",
        report=report,
    )
