"""Vorgerechnete Tabellen (CSV in data/): `python sre_grid.py regeln|raster|horizont|dp|dp2|all` (einige Minuten je Teil, mehrere Prozesse).

- data/regeln.csv     Regeltabelle je Preset (abgestimmt auf 400 Trainingspfaden, Seed 11; ausgewertet auf 2000 getrennten Pfaden, Seed 22)
- data/raster.csv     Raster Resistenzgrad x Fitnesskosten: abgestimmte Schwelle auf 1, 5 und 20 Jahre (Kontrollabstand 2 Wochen; 250 / 1000 Pfade)
- data/horizont.csv   Schwelle und Kalender, abgestimmt auf H Jahre, Summe über 20 Jahre (Grundfall; 250 / 1000 Pfade)
- data/dp_werte.csv   Referenz volle Voraussicht (deterministische Instanz) je Preset: DP, jahresweise bestmögliche Antwort, beste Schwelle, bester Kalender, nie
- data/dp_politik.csv und data/dp2_politik.csv   Politik der Rückwärtsrechnung im Grundfall (ein Wirkstoff, zwei Wirkstoffe), je Jahr
"""

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import sre_constants as C
import sre_dp as DP
import sre_rules as R
from sre_scenario import DEFAULTS, TABLE_KEYS, par_from_settings, settings_of

DATA = Path(__file__).resolve().parent / "data"
REGELN_PATH = DATA / "regeln.csv"
RASTER_PATH = DATA / "raster.csv"
HORIZONT_PATH = DATA / "horizont.csv"
DP_WERTE_PATH = DATA / "dp_werte.csv"
DP_POLITIK_PATH = DATA / "dp_politik.csv"
DP2_POLITIK_PATH = DATA / "dp2_politik.csv"

REGELN_FIELDS = ("scenario",) + TABLE_KEYS + ("rule", "total", "se", "d_nie", "d_nie_se", "d_cal", "d_cal_se", "blk1", "blk2", "blk3", "blk4", "sprays", "resA20", "resB20", "params", "spec")
RASTER_FIELDS = ("phi", "cost", "S1", "S5", "S20", "sum1", "sum5", "sum20", "nie", "cal", "cap", "pause")
HORIZONT_FIELDS = ("H", "S", "v", "sum_thr", "k", "w0", "sum_cal")
DP_WERTE_FIELDS = ("scenario", "dp", "dp_forward", "myopic", "thr", "thr_S", "cal", "cal_k", "cal_w0", "nie", "dp2", "dp2_forward")
DP_POLITIK_FIELDS = ("year", "dp_sprays", "dp_weeks", "dp_res", "my_sprays", "my_weeks", "my_res")
DP2_POLITIK_FIELDS = ("year", "sprays_A", "sprays_B", "res_A", "res_B")


def _write(path, fields, rows):
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def _read(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(x):
    return None if x in ("", None) else float(x)


# ------------------------------------------------------------------------------------------------------------- Regeltabellen
def regeln_scenario(name: str):
    s = settings_of(name)
    par = par_from_settings(s)
    rules = R.tune_rules(par, int(s["horizon"]), C.P_TRAIN, q_ref=float(s["q"]))
    rows = R.rule_table(par, rules, C.P_EVAL, C.SEED_EVAL)
    for row in rows:
        row["scenario"] = name
        row.update({k: s[k] for k in TABLE_KEYS})
    return rows


def build_regeln(workers=6, verbose=True):
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(regeln_scenario, list(C.PRESETS)):
            rows += part
            if verbose:
                print(f"{part[0]['scenario']}: fertig ({time.time() - t0:.0f} s)", flush=True)
    _write(REGELN_PATH, REGELN_FIELDS, rows)
    return rows


def load_regeln(path=REGELN_PATH) -> dict:
    """{Preset-Name: {"settings": {...}, "rows": [dict je Regel in der Reihenfolge von C.RULES]}}; leer, wenn die Datei fehlt."""
    out = {}
    for r in _read(path):
        sc = out.setdefault(r["scenario"], {"settings": {k: float(r[k]) for k in TABLE_KEYS}, "rows": []})
        row = {"rule": r["rule"], "params": r["params"], "spec": r["spec"]}
        for k in ("total", "se", "d_nie", "d_nie_se", "d_cal", "d_cal_se", "blk1", "blk2", "blk3", "blk4", "sprays", "resA20", "resB20"):
            row[k] = _num(r[k])
        sc["rows"].append(row)
    return out


# ------------------------------------------------------------------------------------------------------------------- Raster
def raster_cell(args):
    phi, cost = args
    s = {**DEFAULTS, "phi": phi, "cost": cost}
    par = par_from_settings(s)
    PT, PE = C.P_TRAIN_SWEEP, C.P_EVAL_SWEEP
    out = dict(phi=phi, cost=cost)
    for H, key in ((1, "1"), (5, "5"), (20, "20")):
        b, v, _ = R.tune_v(par, H=H, vs=(2,), P=PT)
        rule = R._rule("S", "thr", b, v=v)
        out["S" + key] = float(b["S"][0])
        out["sum" + key] = float(R.evaluate_rule(par, rule, PE, C.SEED_EVAL)[0].mean())
    from sre_model import never
    out["nie"] = float(R.evaluate_rule(par, R._rule("nie", "cal", never()), PE, C.SEED_EVAL)[0].mean())
    b, _ = R.tune(par, "cal", P=PT)
    out["cal"] = float(R.evaluate_rule(par, R._rule("cal", "cal", b), PE, C.SEED_EVAL)[0].mean())
    b, v, _ = R.tune_v(par, extra=dict(cap=[1, 2, 3, 4, 6]), vs=(2,), P=PT)
    out["cap"] = float(R.evaluate_rule(par, R._rule("cap", "thr", b, v=v), PE, C.SEED_EVAL)[0].mean())
    b, _ = R.tune(par, "cal", grid=R.pause_grid(), P=PT)
    out["pause"] = float(R.evaluate_rule(par, R._rule("pause", "cal", b), PE, C.SEED_EVAL)[0].mean())
    return out


def build_raster(workers=6, verbose=True):
    t0 = time.time()
    cells = [(phi, cost) for phi in C.GRID_PHI for cost in C.GRID_COST]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for row in ex.map(raster_cell, cells):
            rows.append(row)
            if verbose:
                print(f"Raster phi {row['phi']} c {row['cost']}: S1 {row['S1']:.3f} S20 {row['S20']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    _write(RASTER_PATH, RASTER_FIELDS, rows)
    return rows


def load_raster(path=RASTER_PATH) -> list:
    return [{k: float(v) for k, v in r.items()} for r in _read(path)]


def raster_lookup(grid, phi, cost):
    for r in grid:
        if abs(r["phi"] - phi) < 1e-9 and abs(r["cost"] - cost) < 1e-9:
            return r
    return None


# ----------------------------------------------------------------------------------------------------------------- Horizonte
def horizont_row(H):
    par = par_from_settings(DEFAULTS)
    PT, PE = C.P_TRAIN_SWEEP, C.P_EVAL_SWEEP
    b, v, _ = R.tune_v(par, H=H, vs=C.V_SWEEP, P=PT)
    s_sum = float(R.evaluate_rule(par, R._rule("S", "thr", b, v=v), PE, C.SEED_EVAL)[0].mean())
    bc, _ = R.tune(par, "cal", H=H, P=PT)
    c_sum = float(R.evaluate_rule(par, R._rule("cal", "cal", bc), PE, C.SEED_EVAL)[0].mean())
    return dict(H=H, S=float(b["S"][0]), v=v, sum_thr=s_sum, k=int(bc["k"][0]), w0=int(bc["w0"][0]), sum_cal=c_sum)


def build_horizont(workers=6, verbose=True):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(horizont_row, C.HORIZONS))
    if verbose:
        for r in rows:
            print(r, flush=True)
    _write(HORIZONT_PATH, HORIZONT_FIELDS, rows)
    return rows


def load_horizont(path=HORIZONT_PATH) -> list:
    return [{k: float(v) for k, v in r.items()} for r in _read(path)]


# ------------------------------------------------------------------------------------------------------------------------- DP
def dp_values(name: str):
    """Referenz der deterministischen Instanz zu den Einstellungen eines Presets (ein Wirkstoff)."""
    s = settings_of(name)
    par = DP.det_par(par_from_settings(s))
    V, diffs = DP.solve(par)
    v_back = DP.value_at(V, par)
    total, sprays, res, weeks = DP.forward(par, diffs)
    Vm, dm = DP.solve(par, myopic=True)
    tot_m, sp_m, res_m, wk_m = DP.forward(par, dm)
    Ss = np.exp(np.linspace(np.log(0.002), np.log(1.5), 120))
    vals = [DP.threshold_value(par, x) for x in Ss]
    j = int(np.argmax(vals))
    best_cal = max(((DP.calendar_value(par, k, w), k, w) for k in range(1, 27) for w in range(0, min(k, 12))))
    row = dict(scenario=name, dp=v_back, dp_forward=total, myopic=tot_m, thr=vals[j], thr_S=float(Ss[j]), cal=best_cal[0], cal_k=best_cal[1], cal_w0=best_cal[2],
               nie=DP.calendar_value(par, 1000, 1000), dp2="", dp2_forward="")
    policy = [dict(year=y + 1, dp_sprays=sprays[y], dp_weeks=" ".join(map(str, weeks[y])), dp_res=res[y], my_sprays=sp_m[y], my_weeks=" ".join(map(str, wk_m[y])), my_res=res_m[y])
              for y in range(par.Y)]
    return row, policy


# Szenarien mit gleicher Referenz (Kreuzresistenz, Messrauschen, Kontrollkosten, Refugium und Zins gehen in die Referenz nicht ein)
def _dp_key(name):
    s = settings_of(name)
    return tuple(s[k] for k in ("phi", "cost", "kill", "m", "spray"))


def build_dp(workers=4, verbose=True):
    names = list(C.PRESETS)
    keys, first = {}, {}
    for n in names:
        keys.setdefault(_dp_key(n), n)
    uniq = list(keys.values())
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = dict(zip(uniq, ex.map(dp_values, uniq)))
    rows = []
    for n in names:
        row, policy = results[keys[_dp_key(n)]]
        rows.append({**row, "scenario": n})
        if n == names[0]:
            _write(DP_POLITIK_PATH, DP_POLITIK_FIELDS, policy)
    if verbose:
        for r in rows:
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    old = {r["scenario"]: r for r in _read(DP_WERTE_PATH)}
    for r in rows:
        if r["scenario"] in old and old[r["scenario"]].get("dp2"):
            r["dp2"], r["dp2_forward"] = old[r["scenario"]]["dp2"], old[r["scenario"]]["dp2_forward"]
    _write(DP_WERTE_PATH, DP_WERTE_FIELDS, rows)
    return rows


def build_dp2(verbose=True):
    """Zwei Wirkstoffe im Grundfall (rund 8 Minuten). Schreibt die Politik je Jahr und ergänzt den Wert in dp_werte.csv."""
    name = next(iter(C.PRESETS))
    par = DP.det_par2(par_from_settings(settings_of(name)))
    t0 = time.time()
    V, pol = DP.solve2(par)
    v0 = DP.value_at2(V, par)
    total, log = DP.forward2(par, pol)
    if verbose:
        print(f"DP zwei Wirkstoffe: V0 {v0:.1f}, vorwärts {total:.1f} ({time.time() - t0:.0f} s)", flush=True)
    _write(DP2_POLITIK_PATH, DP2_POLITIK_FIELDS, [dict(year=y + 1, sprays_A=a, sprays_B=b, res_A=ra, res_B=rb) for y, (a, b, ra, rb) in enumerate(log)])
    rows = _read(DP_WERTE_PATH)
    for r in rows:
        if r["scenario"] == name:
            r["dp2"], r["dp2_forward"] = v0, total
    _write(DP_WERTE_PATH, DP_WERTE_FIELDS, rows)


def load_dp_werte(path=DP_WERTE_PATH) -> dict:
    out = {}
    for r in _read(path):
        out[r["scenario"]] = {k: (_num(v) if k != "scenario" else v) for k, v in r.items()}
    return out


def load_dp_politik(path=DP_POLITIK_PATH) -> list:
    out = []
    for r in _read(path):
        out.append(dict(year=int(r["year"]), dp_sprays=int(r["dp_sprays"]), dp_weeks=[int(x) for x in r["dp_weeks"].split()], dp_res=float(r["dp_res"]),
                        my_sprays=int(r["my_sprays"]), my_weeks=[int(x) for x in r["my_weeks"].split()], my_res=float(r["my_res"])))
    return out


def load_dp2_politik(path=DP2_POLITIK_PATH) -> list:
    return [dict(year=int(r["year"]), sprays_A=int(r["sprays_A"]), sprays_B=int(r["sprays_B"]), res_A=float(r["res_A"]), res_B=float(r["res_B"])) for r in _read(path)]


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("regeln", "all"):
        build_regeln()
    if what in ("raster", "all"):
        build_raster()
    if what in ("horizont", "all"):
        build_horizont()
    if what in ("dp", "all"):
        build_dp()
    if what in ("dp2", "all"):
        build_dp2()
