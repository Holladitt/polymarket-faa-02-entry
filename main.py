#!/usr/bin/env python3
"""Clean re-implementation of the V7c-jr07 forecast program for the
polymarket-faa-attribution-02 competition (make.fun).

Reads every *.in case in $MAKEFUN_INPUT_DIR (default /input) and writes one
<case>.out per case to $MAKEFUN_OUTPUT_DIR (default /output). For each
market and each of the 240 minutes after the checkpoint it writes five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the Yes price.

Model (re-derived from the design documents; tables fitted from the
reference outputs):
  median = x + deadband(pull + give-back) + d1 + gmom
  fan    = median + z(bucket, tau, h) * sigma(h) * mult(tau, h)
with a board regime scale, a gated pull toward the board sum, an M1
transient give-back, a news-reset halving, and an Iran <= Foreign
projection. Standard library only; deterministic; no network.
"""
import json
import math
import os

# ---------------------------------------------------------------- parameters
KNOTS = [8, 38, 90, 180, 240]

# Effective z table: Z[bucket][tau] = 5 knot values, tau order
# 0.05, 0.25, 0.5, 0.75, 0.95. Bucket by checkpoint price x:
# <0.005, <0.02, <0.10, else.
Z = {
    0: [[-0.3414, -0.4289, -0.4638, -0.4787, -0.6864],
        [-0.0134, -0.0455, -0.0566, -0.0946, -0.1045],
        [0.0001, 0.0004, 0.0001, 0.0005, 0.0169],
        [0.1035, 0.1991, 0.2492, 0.4082, 0.6843],
        [1.7311, 2.435, 2.9726, 2.8577, 3.0508]],
    1: [[-0.9008, -1.2002, -1.1533, -1.836, -1.9545],
        [-0.1403, -0.1745, -0.253, -0.7019, -0.6921],
        [0.0002, -0.0093, -0.0125, -0.0087, -0.1173],
        [0.0901, 0.1661, 0.2604, 0.2609, 0.2668],
        [0.9038, 0.9571, 1.8879, 1.5904, 2.0328]],
    2: [[-1.6908, -1.5569, -1.4673, -1.414, -1.5946],
        [-0.4407, -0.5186, -0.4994, -0.5649, -0.5782],
        [-0.0061, -0.1196, -0.0465, -0.0485, -0.1361],
        [0.0899, 0.2027, 0.3945, 0.6095, 0.5816],
        [0.6826, 1.193, 2.2847, 2.4413, 3.6314]],
    3: [[-1.0251, -1.5153, -0.9572, -1.3231, -1.8299],
        [-0.205, -0.3981, -0.3996, -0.5966, -0.8451],
        [0.0008, -0.042, 0.1678, 0.226, 0.3732],
        [0.4318, 0.8531, 0.587, 0.478, 1.0874],
        [1.7297, 1.8509, 2.1956, 2.4008, 2.4661]],
}

PULL_MAX = 0.931198
PULL_TAU = 161.0
PULL_TAU_FAST = 8.0
K_UP = 0.25
K_DOWN = 1.2
PULL_B = 1.8
REGIME_K = 1.7
VMED = 0.002696
RSCALE_LO = 0.6
RSCALE_HI = 2.5
FAST_GATE_K = 2.7
TARGET_GATE_K = 1.93
STALE_W = 0.70
GB_UP = 0.45
GB_DN = 0.0
GB_TAU = 91.766134
D1_SCALE = 1.5
D1_GATE_K = 3.0
GMOM_K = 0.25
GMOM_TAU = 30.0
GMOM_GATE_K = 2.0
DB_EPS = 0.001175
H5B_DROP = -0.03
H5B_PRICE = 0.08
H5B_MULT = 0.5
TAIL_LIN = 0.218
MID_WIDE = 0.99
# d1 basis knots (horizon) and values.
D1_H = [4.5, 19.5, 60.5, 135.5, 210.5]
D1_V = [0.387, 0.694, -0.368, -0.515, -0.860]
H_SAT = 11.0
YOUNG_BOOST = 1.5
YOUNG_TAU = 50.0


def interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def ewma_sq(returns, halflife, init):
    a = 1.0 - 0.5 ** (1.0 / halflife)
    v = init
    for r in returns:
        v = (1 - a) * v + a * (r * r)
    return v


def sigma1(prices):
    """Per-minute sigma from an EWMA variance blend (fast 15 / slow 240)."""
    n = len(prices)
    rets = []
    for i in range(1, n):
        r = prices[i] - prices[i - 1]
        rets.append(max(-0.06, min(0.06, r)))
    x = prices[-1]
    if not rets:
        return 0.0002 + 0.005 * x
    vf = ewma_sq(rets, 15, 0.0)
    vs = ewma_sq(rets, 240, 0.0)
    s = math.sqrt(0.65 * vf + 0.35 * vs)
    return max(min(s, 0.0102), 0.0002 + 0.005 * x)


def market_feats(m):
    p = m["prices"]
    n = len(p)
    x = p[-1]

    def lag(k):
        return p[-1] - p[max(0, n - 1 - k)] if n >= 2 else 0.0

    r60 = lag(60)
    d1 = p[-1] - p[-2] if n >= 2 else 0.0
    r30 = lag(30)
    r15 = lag(15)
    w = p[-241:] if n >= 241 else p[:]
    zm = 0
    for i in range(1, len(w)):
        if w[i] == w[i - 1]:
            zm += 1
    stale = zm / max(1, len(w) - 1)
    age = 0
    for i in range(n - 1, 0, -1):
        if p[i] != p[i - 1]:
            age = (n - 1) - i
            break
    else:
        age = n - 1
    return dict(x=x, n=n, r60=r60, d1=d1, r30=r30, r15=r15,
                stale=stale, age=age, s1=sigma1(p), prices=p)


def _m1_stats(prices):
    """Shared M1/JR statistics: jump sum J, scale s_t, 60-min move R."""
    n = len(prices)
    rets = [prices[i] - prices[i - 1] for i in range(1, n)]
    sq = [r * r for r in rets]
    warm = sum(sq[:60]) / min(60, len(sq))
    a = 2.0 / 61.0
    s = warm
    s_hist = [0.0] * n
    s_hist[0] = warm
    for i in range(1, n):
        s = (1 - a) * s + a * sq[i - 1]
        s_hist[i] = s
    s_t = max(math.sqrt(s_hist[-1]), 1e-4)
    r = prices[-1] - prices[-61]
    j = 0.0
    for i in range(n - 60, n):
        r_i = prices[i] - prices[i - 1]
        if abs(r_i) > 3 * s_t:
            frac = clip((prices[-1] - prices[i - 1]) / r_i, 0, 1) if r_i != 0 else 0.0
            j += frac * r_i
    return r, j, s_t


def r13_jr(prices):
    """Jump share J/R; 1.0 when R <= 0 or the window is short."""
    n = len(prices)
    if n < 61:
        return 1.0
    r, j, _ = _m1_stats(prices)
    if r <= 0:
        return 1.0
    return j / r if r != 0 else 1.0


def m1_up(prices):
    """Transient (jump-cleaned, continuation-discounted) up move."""
    n = len(prices)
    if n < 61:
        return None
    r, j, s_t = _m1_stats(prices)
    p_t15 = prices[-16] if n >= 16 else prices[0]
    c_raw = (1 if r >= 0 else -1) * (prices[-1] - p_t15) / max(0.25 * abs(r), s_t)
    c = clip(c_raw, 0, 1)
    r_tr = (r - j) * (1 - 0.8 * c)
    return max(r_tr, 0.0)


def bucket_x(x):
    if x < 0.005:
        return 0
    if x < 0.02:
        return 1
    if x < 0.10:
        return 2
    return 3


def sigma_h(s1, n, h, fscale):
    young = 1 + YOUNG_BOOST * math.exp(-n / YOUNG_TAU)
    return s1 * math.sqrt(h) / math.sqrt(1 + h / H_SAT) * young * fscale


def forecast_case(case):
    mkts = case["markets"]
    feats = [market_feats(m) for m in mkts]
    xs = [f["x"] for f in feats]
    ssum = sum(xs)
    gap = 1.0 - ssum
    # Board regime: mean sigma1 drives the pull scale and the fan scale.
    v = sum(f["s1"] for f in feats) / len(feats)
    rscale = clip(1 + REGIME_K * (v - VMED) / VMED, RSCALE_LO, RSCALE_HI)
    fscale = clip((v / VMED) ** 0.25, 0.5, 2.0)
    pb = 2.25 if rscale < 1.2 else PULL_B
    wts = []
    for f in feats:
        wts.append(f["x"] * (f["s1"] ** pb) *
                   (f["stale"] ** STALE_W if f["stale"] > 0 else 0.0))
    wsum = sum(wts)
    gate_pull = 1.0 if (len(mkts) >= 10 and ssum > 0.5) else 0.0
    g_lin = clip(TARGET_GATE_K * (rscale - 1), 0, 1)
    w_fast = clip(FAST_GATE_K * (rscale - 1), 0, 1)
    g_gmom = clip(GMOM_GATE_K * (rscale - 1), 0, 1)
    d1_gate = clip(D1_GATE_K * (rscale - 1), 0, 1)
    # Structural board gate for the M1 transient mechanism.
    g_board = 1 if (19 <= len(mkts) <= 24 and 0.81 <= ssum <= 1.09) else 0
    k = K_UP if gap > 0 else K_DOWN

    names = [m["name"] for m in mkts]
    i_iran = next((i for i, nm in enumerate(names)
                   if "iran" in nm.lower() and "foreign" not in nm.lower()), None)
    i_for = next((i for i, nm in enumerate(names)
                  if "foreign" in nm.lower()), None)

    out = []
    for idx, (m, f) in enumerate(zip(mkts, feats)):
        x = f["x"]
        adj_lin = x / ssum - x if ssum > 0 else 0.0
        adj_w = gap * (wts[idx] / wsum) if wsum > 0 else 0.0
        adj = (1 - g_lin) * adj_w + g_lin * adj_lin
        # News reset: a sharp 60-minute drop on a cheap market halves drift.
        h5b = H5B_MULT if (f["r60"] <= H5B_DROP and x < H5B_PRICE) else 1.0
        # M1 transient give-back, only under the board gate and J/R < 0.9.
        up_m1 = None
        if g_board and f["n"] >= 61:
            if r13_jr(f["prices"]) < 0.9:
                up_m1 = m1_up(f["prices"])
        b = bucket_x(x)
        pin = clip(1.0 - f["age"] / 120.0, 0.0, 1.0)
        rows = []
        for h in range(1, 241):
            ramp_s = 1 - math.exp(-h / PULL_TAU)
            ramp_f = 1 - math.exp(-h / PULL_TAU_FAST)
            pull = gate_pull * k * adj * PULL_MAX * rscale * (ramp_s + w_fast * ramp_f)
            up = up_m1 if up_m1 is not None else max(f["r60"], 0.0)
            dn = min(f["r60"], 0.0)
            w_gb = 1 - math.exp(-h / GB_TAU)
            gb = -(GB_UP * up + GB_DN * dn) * w_gb
            drift = (pull + gb) * h5b
            base = x if abs(drift) < DB_EPS else x + drift
            d1t = D1_SCALE * interp(h, D1_H, D1_V) * f["d1"] * d1_gate * h5b
            gm = 0.0
            if f["r30"] < 0:
                gm = g_gmom * GMOM_K * f["r30"] * (1 - math.exp(-h / GMOM_TAU)) * h5b
            med = base + d1t + gm
            db = 0.9 if abs(drift) < DB_EPS else 1.0
            s = sigma_h(f["s1"], f["n"], h, fscale)
            # Median (q50) carries the z50 shift from the latent median;
            # the other quantiles fan out from q50.
            z50 = interp(h, KNOTS, Z[b][2])
            q50 = med + z50 * s * db
            row = [0.0] * 5
            row[2] = clip(q50, 0.0, 1.0)
            for tau in (0, 1, 3, 4):
                z = interp(h, KNOTS, Z[b][tau])
                if tau in (0, 4):
                    mult = (1.0 + TAIL_LIN * h / 240.0) * db
                else:
                    mult = MID_WIDE * pin * db
                row[tau] = clip(q50 + z * s * mult, 0.0, 1.0)
            rows.append(row)
        out.append({"id": m["id"], "q": rows})

    # Projection: Iran <= Foreign on every quantile cell.
    if i_iran is not None and i_for is not None:
        qi = out[i_iran]["q"]
        qf = out[i_for]["q"]
        for h in range(240):
            for tau in range(5):
                if qi[h][tau] > qf[h][tau]:
                    qi[h][tau] = qf[h][tau]

    # Round to 6 decimals and enforce non-decreasing rows.
    for fc in out:
        for h in range(240):
            row = fc["q"][h]
            row = [round(v, 6) for v in row]
            for tau in range(1, 5):
                if row[tau] < row[tau - 1]:
                    row[tau] = row[tau - 1]
            fc["q"][h] = row
    return {"forecasts": out}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for fn in sorted(os.listdir(in_dir)):
        if not fn.endswith(".in"):
            continue
        with open(os.path.join(in_dir, fn)) as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = fn[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w") as fh:
            json.dump(result, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
