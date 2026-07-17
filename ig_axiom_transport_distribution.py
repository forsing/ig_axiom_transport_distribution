from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
Aksiom + next iz GLOBALNE DISTRIBUCIJE (ne frekvencija, ne transport).

Razlika:
  frekvencija  → rang po count_i (marginalni broj)
  distribucija → zakon na simpleksu p_glob + zona-po-kolu iz CSV
                 (empirijska struktura kola, ne geodezija/OT/Monge)

1) Provera: lid ⇒ ¬transport_predictive  (kao ig_axiom_transport_memory)
2) next: 7 brojeva čija μ=ind/7 najbolje pogađa p_glob i zone-profil kola
   ban last. Bez transport skora.

CSV: loto7_4650_k56.csv, seed=39.
Ime: ig_axiom_transport_distribution.py
"""

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from ig_axiom_transport_memory import (
    lid_process_empirical,
    memory_empirical,
    path_series,
    transport_exist,
    transport_existence_requires_memory,
    transport_predictive,
)

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4650_k56.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def zone(n: int) -> int:
    if n <= 13:
        return 0
    if n <= 26:
        return 1
    return 2


def global_distribution(draws: np.ndarray) -> np.ndarray:
    """p_glob na simpleksu Δ^38 — marginalna distribucija, ne sirovi count."""
    cnt = Counter(draws.reshape(-1).tolist())
    n_slots = len(draws) * FRONT_SELECT
    p = np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)
    return p / p.sum()


def zone_profile_per_draw(draws: np.ndarray) -> np.ndarray:
    """Prosek (n_low, n_mid, n_high) po kolu — distribucija zone-strukture."""
    prof = np.zeros(3)
    for d in draws:
        z = [0, 0, 0]
        for x in d.tolist():
            z[zone(int(x))] += 1
        prof += np.array(z, dtype=float)
    return prof / float(len(draws))


def combo_measure(combo: list[int]) -> np.ndarray:
    mu = np.zeros(FRONT_N)
    for x in combo:
        mu[int(x) - 1] = 1.0 / FRONT_SELECT
    return mu


def combo_zone_vec(combo: list[int]) -> np.ndarray:
    z = np.zeros(3)
    for x in combo:
        z[zone(int(x))] += 1.0
    return z


def distribution_score(
    combo: list[int],
    p_glob: np.ndarray,
    zone_target: np.ndarray,
    ban: set[int],
) -> float:
    if any(x in ban for x in combo):
        return -1e18
    mu = combo_measure(combo)
    l2 = float(np.linalg.norm(mu - p_glob))
    z = combo_zone_vec(combo)
    z_err = float(np.linalg.norm(z - zone_target))
    return -(l2 + 0.15 * z_err)


def number_scores(
    p_glob: np.ndarray,
    zone_target: np.ndarray,
    ban: set[int],
) -> dict[int, float]:
    """
    Doprinos broja n globalnoj distribuciji (excess na simpleksu)
    + blagi zone-fit (ne frekvencijski rang).
    """
    p0 = 1.0 / FRONT_N
    z_share = zone_target / zone_target.sum()
    out = {}
    for i in range(FRONT_N):
        n = i + 1
        if n in ban:
            out[n] = -1e18
        else:
            out[n] = float((p_glob[i] - p0) + 0.2 * z_share[zone(n)])
    return out


def _combo_fit(combo, score, target_sum, pos_means, target_odd, ban, p_glob, zone_target):
    nums = sorted(combo)
    base = distribution_score(nums, p_glob, zone_target, ban)
    if base < -1e17:
        return base
    s = base + 0.5 * sum(score[x] for x in nums)
    s -= 0.08 * abs(sum(nums) - target_sum)
    s -= 0.04 * sum(abs(nums[i] - pos_means[i]) for i in range(FRONT_SELECT))
    odd = sum(1 for x in nums if x % 2)
    s -= 0.3 * abs(odd - target_odd)
    return s


def predict_next(draws, p_glob, zone_target, ban):
    score = number_scores(p_glob, zone_target, ban)
    ranked = sorted((n for n in score if n not in ban), key=lambda n: (-score[n], n))
    target_sum = float(draws.sum(axis=1).mean())
    pos_means = [float(draws[:, i].mean()) for i in range(FRONT_SELECT)]
    target_odd = float(np.mean([sum(1 for x in d if x % 2) for d in draws]))
    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, len(ranked) - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))
    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, target_sum, pos_means, target_odd, ban, p_glob, zone_target)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(
                    cand, score, target_sum, pos_means, target_odd, ban, p_glob, zone_target
                )
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_distribution(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    last = draws[-1]
    ban = set(int(x) for x in last.tolist())
    ps = path_series(draws)
    mem = memory_empirical(ps)
    lid = lid_process_empirical(ps)
    tex = transport_exist(ps[-2], ps[-1]) if len(ps) >= 2 else False
    tpred = transport_predictive(ps)
    axiom_ok = transport_existence_requires_memory(lid, tpred)

    p_glob = global_distribution(draws)
    zone_target = zone_profile_per_draw(draws)
    combo = predict_next(draws, p_glob, zone_target, ban)

    print(f"CSV: {csv_path.name}")
    print(f"Kola: {len(draws)} | seed={SEED} | ig_axiom_transport_distribution")
    print(f"last: {last.tolist()}")
    print()
    print("=== aksiom (i.i.d. → nema prediktivnog transporta) ===")
    print(
        {
            "memory_empirical": mem,
            "lid_process_empirical": lid,
            "transport_exist_formal": tex,
            "transport_predictive": tpred,
            "axiom_ok": axiom_ok,
        }
    )
    print()
    print("=== globalna distribucija ===")
    print(
        {
            "zone_target_low_mid_high": [round(float(x), 4) for x in zone_target],
            "p_glob_l2_uniform": round(float(np.linalg.norm(p_glob - 1 / FRONT_N)), 6),
        }
    )
    print()
    ranked = sorted(
        ((n, float(p_glob[n - 1])) for n in range(1, FRONT_N + 1) if n not in ban),
        key=lambda t: (-t[1], t[0]),
    )
    print("=== top8 po p_glob (distribucija, ne transport) ===")
    print([(n, round(p, 6)) for n, p in ranked[:8]])
    print()
    print("=== next (distribucija, bez transporta) ===")
    print("next:", combo)


if __name__ == "__main__":
    run_distribution()



"""
CSV: loto7_4650_k56.csv
Kola: 4650 | seed=39 | ig_axiom_transport_distribution
last: [4, 5, 6, 11, 12, 18, 28]

=== aksiom (i.i.d. → nema prediktivnog transporta) ===
{'memory_empirical': True, 'lid_process_empirical': False, 'transport_exist_formal': False, 'transport_predictive': True, 'axiom_ok': True}

=== globalna distribucija ===
{'zone_target_low_mid_high': [2.3417, 2.3191, 2.3391], 'p_glob_l2_uniform': 0.006037}

=== top8 po p_glob (distribucija, ne transport) ===
[(8, 0.028111), (23, 0.027896), (34, 0.026912), (26, 0.026759), (37, 0.026513), (32, 0.026421), (33, 0.026298), (29, 0.026237)]

=== next (distribucija, bez transporta) ===
next: [3, x, 13, y, 24, z, 38]
"""



"""
aksiom + provera (import iz ig_axiom_transport_memory)
next iz p_glob + zone-profil kola (distribucija na simpleksu), ne transport
"""
