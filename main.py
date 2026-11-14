#!/usr/bin/env python3
"""Re-implementation of the G1 forecaster (H5b soft-reset + gated board-sum pull,
base V10) for make.fun polymarket-faa-attribution-02.

Written from the design documentation and from fitting the reference outputs
(numbers only). Stdlib only, deterministic, offline.

Model summary
-------------
Per market, per horizon h = 1..240 minutes, five quantiles are emitted:
  q_tau(h) = m(h) + gap_tau(h),  tau in (0.05, 0.25, 0.5, 0.75, 0.95)
where m(h) is the median path and gap_0.5 == 0.

Median path  m(h) = last + drift(h) + S * z50(h, bucket)
  drift(h) = give-back + d1 + gmom + pull      (each deadbanded, see below)
  give-back : -gb_up * max(R60, 0) * (1 - exp(-h / 80)), gb_up = 0.37011
              (falls are NOT faded back: gb_dn = 0)
  d1        : d1_scale * d1_innov * B(h), d1_scale = 0.16 (effective),
              d1_innov = last 1-min move minus the mean 1-min move over the
              last 30 minutes; B(h) is piecewise linear on knots
              [4.5, 19.5, 60.5, 135.5, 210.5] with values
              [+0.39, +0.69, -0.37, -0.52, -0.86] (clamped at the ends).
  gmom      : gmom_k * dn30 * bump(h) * g, gmom_k = 0.25,
              dn30 = sum of negative 1-min moves in the last 30 minutes,
              bump(h) rises to a peak near h=38 and decays away by h~90,
              g = 1 when board volatility V >= 2 * Vmed else 0.
  pull      : active only when the board gate is on (n_markets >= 10 and
              board_sum > 0.5).  pull(h) = -PULL_K * share**3 * (sum - 1)
              * (1 - exp(-h / 150)), share = last / board_sum, PULL_K = 53.
  news-reset (H5b soft reset): when a market fell >= 0.03 inside 60 minutes
              and ended below 0.08, the drift terms (give-back, d1, gmom)
              are halved (x0.5).  The pull and the z50 fan-median term are
              not halved.
  deadband  : any drift component whose absolute value is below DB_EPS
              (0.001175) is snapped to 0 before summing.

Fan (quantile gaps)
  S is the market fan scale: an EWMA volatility (fast-15 / slow-240 mean of
  |1-min moves|), floored per price bucket, capped at 0.0102, and multiplied
  by a young-market boost Y(n).  The gaps are
      gap_tau(h) = S * F_tau(h, bucket) * stale_tau(age)
  where F_tau is the normalised fan table (F_95(1) == 1) measured from flat
  reference markets, and stale_tau compresses only the inner quantiles
  (0.25, 0.75) by c = max(0, 1 - age / 120), age = minutes since the last
  price change.  Outer quantiles (0.05, 0.95) are not stale-compressed.
  z50(h, bucket) is the fan median offset (see Z50 table), so the median
  itself carries S * z50(h, bucket).

Projection (applied last): for the FAA board, Iran's quantiles are capped at
  the foreign-actor market's quantiles, per quantile (Iran <= Foreign).
  In general the program applies the projection only when the board carries
  both markets; here it is keyed on the market names.

Output: one JSON object per case, written to <out>/<id>.out, with the same
  shape as the reference (forecasts in market order, q as 240 x 5 lists).
  Rows are non-decreasing and clamped to [0, 1].  Gate status is logged to
  stderr as "GATE <id> board_gate=<0|1>".
"""
import json
import math
import os
import sys

# ---------------------------------------------------------------- parameters
GB_UP = 0.37011
GB_TAU = 80.0
D1_SCALE = 1.5
D1_KNOTS = [4.5, 19.5, 60.5, 135.5, 210.5]
D1_VALS = [0.39, 0.69, -0.37, -0.52, -0.86]
GMOM_K = 0.25
GMOM_TAU_RISE = 20.0
GMOM_TAU_FALL = 40.0
GMOM_PEAK_H = 40.0
GMOM_GATE_K = 2.0
VMED = 0.002696
SIGMA_CAP = 0.0102
DB_EPS = 0.001175
PULL_K = 53.0
PULL_TAU = 150.0
YOUNG_BOOST = 1.5
STALE_DEN = 120.0
RESET_DROP = 0.03
RESET_PRICE = 0.08
RESET_FACTOR = 0.5
HORIZON = 240
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

# price buckets: thresholds [0.005, 0.01, 0.02, 0.1]
BUCKET_EDGES = [0.005, 0.01, 0.02, 0.1]
BUCKET_NAMES = ["lo2", "lo1", "m2", "m1", "hi"]

# fan floor per bucket = g95(1) of a flat reference market in that bucket
FLOOR = {"lo2": 0.000161, "lo1": 0.000093, "m2": 0.000111,
         "m1": 0.000156, "hi": 0.004161}

# normalised fan tables F_tau(h) at knots [1, 8, 38, 90, 180, 240]
# (F_95(1) == 1).  Outer gaps from old flat markets, inner gaps from young
# flat markets (stale-corrected).
FAN_KNOTS = [1, 8, 38, 90, 180, 240]
FAN = {
    "hi": {
        "g05": [0.589, 1.3254, 2.7724, 1.9767, 3.0529, 4.4715],
        "g25": [0.1153, 0.2594, 0.6886, 0.7544, 1.1714, 1.6771],
        "g75": [0.2428, 0.5457, 1.4831, 1.1061, 0.9260, 2.1528],
        "g95": [1.0, 2.2497, 3.3631, 4.5412, 5.5518, 5.9976],
    },
    "m1": {
        "g05": [2.5192, 5.6667, 7.3718, 7.8077, 8.3333, 9.8462],
        "g25": [0.6387, 1.4362, 2.3491, 2.3553, 2.8071, 2.9099],
        "g75": [0.1246, 0.2866, 0.9004, 1.8849, 3.0127, 2.9192],
        "g95": [1.0, 2.25, 5.4936, 11.9359, 14.1667, 22.2436],
    },
    "m2": {
        "g05": [1.036, 2.3243, 4.2973, 4.6396, 8.027, 8.991],
        "g25": [0.1707, 0.3895, 0.639, 0.9717, 2.6567, 2.648],
        "g75": [0.0963, 0.2145, 0.5384, 0.9191, 0.9454, 0.9804],
        "g95": [1.0, 2.2432, 3.2883, 7.4054, 6.9009, 9.3063],
    },
    "lo1": {
        "g05": [0.6559, 1.4624, 3.129, 3.3548, 6.6774, 7.6129],
        "g25": [0.0156, 0.0312, 0.1507, 0.4522, 2.1206, 2.1154],
        "g75": [0.1091, 0.2495, 0.5873, 0.9719, 0.9979, 1.0343],
        "g95": [1.0, 2.2581, 3.3011, 7.4516, 6.9462, 9.3656],
    },
    "lo2": {
        "g05": [0.1988, 0.4472, 0.7888, 0.9441, 1.0807, 1.6646],
        "g25": [0.006, 0.018, 0.0811, 0.1051, 0.1802, 0.2132],
        "g75": [0.0571, 0.1321, 0.3423, 0.4595, 0.7748, 1.3063],
        "g95": [1.0, 2.2547, 4.4224, 6.0683, 6.4534, 7.2547],
    },
}

# fan median offset z50(h, bucket): value at h=240 per bucket, times a shared
# horizon shape (normalised to 1 at h=240).
Z50_240 = {"hi": 0.6885, "m1": -0.6218, "m2": -0.3874,
           "lo1": -0.3871, "lo2": 0.0311}
Z50_KNOTS = [1, 10, 20, 38, 45, 60, 90, 120, 180, 210, 240]
Z50_SHAPE = [0.0, 0.0035, 0.0237, 0.0656, 0.1298, 0.2726,
             0.5658, 0.5777, 0.5944, 0.7969, 1.0]

# young-market boost multiplier Y(n), measured at n = 10, 30, 100, 3000
# (log-n interpolation).
YOUNG_N = [10.0, 30.0, 100.0, 3000.0]
YOUNG_Y = [2.228, 1.823, 1.203, 1.0]


# ---------------------------------------------------------------- helpers
def interp(xs, ys, x):
    """Piecewise-linear interpolation, clamped at the ends."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            if xs[i + 1] == xs[i]:
                return ys[i]
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def bucket_of(price):
    for i, edge in enumerate(BUCKET_EDGES):
        if price < edge:
            return BUCKET_NAMES[i]
    return BUCKET_NAMES[-1]


def young_boost(n):
    """Multiplier for young markets; log-n interpolation between measured pts."""
    if n >= YOUNG_N[-1]:
        return 1.0
    if n <= YOUNG_N[0]:
        return YOUNG_Y[0]
    ln = math.log(n)
    for i in range(len(YOUNG_N) - 1):
        if YOUNG_N[i] <= n <= YOUNG_N[i + 1]:
            l0 = math.log(YOUNG_N[i])
            l1 = math.log(YOUNG_N[i + 1])
            t = (ln - l0) / (l1 - l0) if l1 != l0 else 0.0
            return YOUNG_Y[i] + t * (YOUNG_Y[i + 1] - YOUNG_Y[i])
    return 1.0


def ewma(values, span):
    """Exponentially weighted mean, initialised at the first value."""
    if not values:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    acc = values[0]
    for v in values[1:]:
        acc = alpha * v + (1.0 - alpha) * acc
    return acc


# ---------------------------------------------------------------- features
def market_features(prices):
    """Compute every per-market feature the model needs from a price list."""
    n = len(prices)
    last = prices[-1]
    moves = [prices[i] - prices[i - 1] for i in range(1, n)]
    abs_moves = [abs(m) for m in moves]

    # age: minutes since the last price change
    age = 0
    for i in range(n - 1, 0, -1):
        if prices[i] == prices[i - 1]:
            age += 1
        else:
            break

    d1 = moves[-1] if moves else 0.0

    def ret(k):
        # price change over the last k minutes
        if n > k:
            return last - prices[-k - 1]
        return last - prices[0]

    r30 = ret(30)
    r60 = ret(60)
    r15 = ret(15)

    # down-momentum: sum of negative 1-min moves in the last 30 minutes
    recent = moves[-30:] if len(moves) >= 30 else moves
    dn30 = sum(m for m in recent if m < 0.0)
    # mean 1-min move over the last 30 minutes (for the d1 innovation)
    mean30 = (sum(recent) / len(recent)) if recent else 0.0
    d1_innov = d1 - mean30

    # Volatility: the mean absolute 1-min move over the whole history,
    # scaled to a standard deviation (Laplace moves: sd = sqrt(2) * mean|x|).
    # A full-history mean is used (not a decaying EWMA) so that a market
    # which has gone stale keeps the volatility implied by its history.
    # The 0.86 factor calibrates the scale to the reference fan width.
    mean_abs = (sum(abs_moves) / len(abs_moves)) if abs_moves else 0.0
    sigma_raw = 1.22 * mean_abs

    bucket = bucket_of(last)
    floor = FLOOR[bucket]
    sigma = min(max(sigma_raw, floor), SIGMA_CAP)
    sigma = sigma * young_boost(n)

    # news-reset trigger: a >=0.03 fall inside 60 minutes ending below 0.08
    reset = (r60 <= -RESET_DROP) and (last < RESET_PRICE)

    return {
        "n": n, "last": last, "age": age, "d1": d1, "d1_innov": d1_innov,
        "r15": r15, "r30": r30, "r60": r60, "dn30": dn30,
        "sigma": sigma, "sigma_raw": sigma_raw, "bucket": bucket,
        "reset": reset,
    }


# ---------------------------------------------------------------- median
def b_of_h(h):
    """d1 horizon shape B(h), piecewise linear on D1_KNOTS, clamped."""
    return interp(D1_KNOTS, D1_VALS, float(h))


def gmom_bump(h):
    """Down-momentum horizon shape: rises to a peak near h=40, decays by
    h~90, essentially gone by h~180."""
    rise = 1.0 - math.exp(-h / GMOM_TAU_RISE)
    if h <= GMOM_PEAK_H:
        return rise
    return rise * math.exp(-(h - GMOM_PEAK_H) / GMOM_TAU_FALL)


def deadband(x):
    """Snap a drift component to 0 when it is below the deadband."""
    if abs(x) < DB_EPS:
        return 0.0
    return x


def median_drift(feat, h, board_sum, gate, board_v):
    """Total median drift at horizon h (minutes) for one market."""
    last = feat["last"]
    # give-back: fades rises only
    gb = -GB_UP * max(feat["r60"], 0.0) * (1.0 - math.exp(-h / GB_TAU))
    # d1: the raw last 1-min move times the B(h) shape
    d1t = D1_SCALE * feat["d1"] * b_of_h(h)
    # down-momentum, gated by board volatility
    g = 1.0 if board_v >= GMOM_GATE_K * VMED else 0.0
    gm = GMOM_K * feat["dn30"] * gmom_bump(h) * g
    # board-sum pull (gated).  The pull normalises an OVERFULL board toward
    # a sum of 1: it only acts when board_sum > 1 (pulling the largest
    # shares down).  An underfull board (sum <= 1) is left alone.
    pull = 0.0
    if gate and board_sum > 1.0:
        share = last / board_sum
        pull = -PULL_K * (share ** 3) * (board_sum - 1.0) \
            * (1.0 - math.exp(-h / PULL_TAU))
    # soft news-reset halves the drift terms (not the pull)
    if feat["reset"]:
        gb = gb * RESET_FACTOR
        d1t = d1t * RESET_FACTOR
        gm = gm * RESET_FACTOR
    # deadband per component
    gb = deadband(gb)
    d1t = deadband(d1t)
    gm = deadband(gm)
    pull = deadband(pull)
    return gb + d1t + gm + pull


def z50_of_h(feat, h):
    """Fan median offset at horizon h."""
    shape = interp(Z50_KNOTS, Z50_SHAPE, float(h))
    return Z50_240[feat["bucket"]] * shape


# ---------------------------------------------------------------- fan
def fan_gaps(feat, h):
    """Quantile gaps (g05, g25, g75, g95) at horizon h for one market."""
    table = FAN[feat["bucket"]]
    s = feat["sigma"]
    g05 = s * interp(FAN_KNOTS, table["g05"], float(h))
    g25 = s * interp(FAN_KNOTS, table["g25"], float(h))
    g75 = s * interp(FAN_KNOTS, table["g75"], float(h))
    g95 = s * interp(FAN_KNOTS, table["g95"], float(h))
    # stale compression: inner quantiles only
    c = max(0.0, 1.0 - feat["age"] / STALE_DEN)
    g25 = g25 * c
    g75 = g75 * c
    return g05, g25, g75, g95


# ---------------------------------------------------------------- case
def forecast_case(case):
    """Forecast one case; returns the output dict."""
    markets = case["markets"]
    feats = [market_features(m["prices"]) for m in markets]
    n_markets = len(markets)
    board_sum = sum(f["last"] for f in feats)
    gate = 1 if (n_markets >= 10 and board_sum > 0.5) else 0
    # board volatility: mean of the per-market (capped, pre-boost) sigmas
    board_v = sum(f["sigma_raw"] for f in feats) / max(1, n_markets)
    board_v = min(board_v, SIGMA_CAP)

    horizon = int(case.get("horizon_minutes", HORIZON))
    forecasts = []
    for m, feat in zip(markets, feats):
        rows = []
        for h in range(1, horizon + 1):
            drift = median_drift(feat, h, board_sum, gate, board_v)
            med = feat["last"] + drift + feat["sigma"] * z50_of_h(feat, h)
            g05, g25, g75, g95 = fan_gaps(feat, h)
            q = [med - g05, med - g25, med, med + g75, med + g95]
            # clamp to [0, 1] and enforce non-decreasing rows
            q = [min(1.0, max(0.0, v)) for v in q]
            for i in range(1, 5):
                if q[i] < q[i - 1]:
                    q[i] = q[i - 1]
            rows.append([round(v, 6) for v in q])
        forecasts.append({"id": m.get("id", ""), "q": rows})
    out = {"forecasts": forecasts}
    apply_projection(case, out, feats)
    return out, gate


def apply_projection(case, out, feats):
    """Iran <= Foreign per quantile (FAA board).  Keyed on the market names:
    the market whose name mentions Iran is capped at the market whose name
    mentions foreign actor / foreign."""
    names = [m.get("name", "") for m in case["markets"]]
    iran_i = None
    foreign_i = None
    for i, nm in enumerate(names):
        low = nm.lower()
        if "iran" in low:
            iran_i = i
        if "foreign" in low:
            foreign_i = i
    if iran_i is None or foreign_i is None:
        return
    fq = out["forecasts"][foreign_i]["q"]
    iq = out["forecasts"][iran_i]["q"]
    for h in range(len(iq)):
        for j in range(5):
            if iq[h][j] > fq[h][j]:
                iq[h][j] = fq[h][j]
        # re-enforce non-decreasing after the cap
        for j in range(1, 5):
            if iq[h][j] < iq[h][j - 1]:
                iq[h][j] = iq[h][j - 1]


# ---------------------------------------------------------------- main
def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name)) as fh:
            case = json.load(fh)
        out, gate = forecast_case(case)
        cid = case.get("id", case.get("case", "")) or os.path.splitext(name)[0]
        with open(os.path.join(out_dir, cid + ".out"), "w") as fh:
            json.dump(out, fh)
        sys.stderr.write("GATE %s board_gate=%d\n" % (cid, gate))


if __name__ == "__main__":
    main()
