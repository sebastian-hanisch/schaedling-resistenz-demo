"""Regeln, Abstimmung und Auswertung.

Jede Regel wird auf Trainingspfaden abgestimmt (Zufalls-Seed 11) und auf GETRENNTEN Auswertungspfaden gemessen (Seed der Regler, Standard 22);
alle Regeln sehen dieselben Auswertungspfade (gepaarte Unterschiede, Standardfehler der Unterschiede werden mit ausgewiesen). Kein Strohmann:
jede Regel ist abgestimmt; die „blind“ auf 1 Jahr abgestimmte Schwelle ist ausdrücklich als Kurzsicht-Fall benannt."""

import itertools
import json
from dataclasses import dataclass

import numpy as np

import sre_constants as C
from sre_model import Par, mean_se, never, simulate, total, with_


@dataclass(frozen=True)
class Rule:
    name: str
    kind: str                # "cal" (Kalender) oder "thr" (Schwelle)
    mode: str = "A"          # Wirkstoffwahl, siehe sre_model.simulate
    prm: tuple = ()          # ((Name, Wert), ...): k, w0 | S, cap, per, Ysw
    v: int = 1               # Kontrollabstand in Wochen (nur Schwelle)
    q: float = 0.0           # Refugium-Anteil, mit dem die Regel ausgewertet wird

    def arrays(self) -> dict:
        return {k: np.array([val]) for k, val in self.prm}

    def get(self, key, default=None):
        return dict(self.prm).get(key, default)

    def describe(self) -> str:
        labels = {"k": "alle {} Wochen", "w0": "ab Woche {}", "cap": "höchstens {} je Saison", "per": "nur jedes {}. Jahr", "Ysw": "A in Jahr 1 bis {}, danach B"}
        prm = dict(self.prm)
        parts = []
        for k, val in self.prm:
            if k == "S":
                parts.append(f"S = {val:.3g}")
            elif k == "k" and val >= 1000:
                parts.append("nie")
            elif k == "w0" and prm.get("k", 0) >= 1000:
                continue
            elif k in labels:
                parts.append(labels[k].format(int(val)))
        if self.kind == "thr":
            parts.append(f"Kontrolle alle {self.v} Wochen")
        if self.q > 0:
            parts.append(f"Refugium {int(round(100 * self.q))} %")
        return ", ".join(parts)

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, "mode": self.mode, "prm": dict(self.prm), "v": self.v, "q": self.q}, sort_keys=True)

    @staticmethod
    def from_json(name: str, text: str) -> "Rule":
        d = json.loads(text)
        return Rule(name, d["kind"], d["mode"], tuple(sorted(d["prm"].items())), int(d["v"]), float(d["q"]))


# ----------------------------------------------------------------------------------------------------- Gitter der Abstimmung
def s_grid(step=1) -> np.ndarray:
    full = np.exp(np.linspace(np.log(C.S_GRID_LO), np.log(C.S_GRID_HI), C.S_GRID_N))
    return full[::step].copy()


def cal_grid():
    """Kalender (k, w0): k = 1 bis 13, 16, 20, 26 mit Startwochen 0 bis min(k, 8) − 1, dazu „nie“ (k = 1000)."""
    ks, w0s = [], []
    for k in list(range(1, 14)) + [16, 20, 26]:
        for w0 in range(0, min(k, 8)):
            ks.append(k)
            w0s.append(w0)
    ks.append(1000)
    w0s.append(1000)
    return dict(k=np.array(ks), w0=np.array(w0s))


def pause_grid():
    """Pausenkalender: Wochenabstand k, Start, höchstens `cap` Spritzungen je Saison, nur in jedem `per`-ten Jahr."""
    ks, w0, cap, per = [], [], [], []
    for k, w, c, p in itertools.product([1, 2, 4], [0, 2, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 6]):
        ks.append(k)
        w0.append(w)
        cap.append(c)
        per.append(p)
    return dict(k=np.array(ks), w0=np.array(w0), cap=np.array(cap), per=np.array(per))


def _product(grid: dict, extra: dict) -> dict:
    names = list(extra)
    combos = list(itertools.product(*[extra[n] for n in names]))
    G0 = len(next(iter(grid.values())))
    out = {k: np.repeat(a, len(combos)) for k, a in grid.items()}
    for j, n in enumerate(names):
        out[n] = np.tile(np.array([c[j] for c in combos]), G0)
    return out


# ----------------------------------------------------------------------------------------------------- Abstimmung
def tune(par: Par, kind: str, mode="A", H=None, extra=None, P=C.P_TRAIN, seed=C.SEED_TRAIN, v=1, grid=None, noise=True):
    """Beste Parameter auf den Trainingspfaden. H = Zahl der Jahre, über die summiert wird (Standard par.Y); der Zinsfaktor von `par` zinst ab.
    Rückgabe: (beste Parameter als dict mit Arrays der Länge 1, Wert auf den Trainingspfaden)."""
    prm = grid if grid is not None else (cal_grid() if kind == "cal" else dict(S=s_grid()))
    if extra:
        prm = _product(prm, extra)
    res = simulate(par, kind, prm, P, seed, years=H, mode=mode, noise=noise, v=v)
    tot = total(res, par, H).mean(1)
    i = int(np.argmax(tot))
    return {k: a[i:i + 1] for k, a in prm.items()}, float(tot[i])


def tune_v(par: Par, mode="A", H=None, extra=None, grid=None, vs=C.V_OPTIONS, **kw):
    """Schwelle mit abgestimmtem Kontrollabstand. Rückgabe: (beste Parameter, v, Wert auf den Trainingspfaden)."""
    best = None
    for v in vs:
        b, t = tune(par, "thr", mode=mode, H=H, extra=extra, grid=grid, v=v, **kw)
        if best is None or t > best[2]:
            best = (b, v, t)
    return best


def _rule(name, kind, best, mode="A", v=1, q=0.0) -> Rule:
    return Rule(name, kind, mode, tuple((k, float(a[0])) for k, a in best.items()), v, q)


def tune_rules(par: Par, H: int, p_train: int, q_ref: float = C.DEFAULT_Q, vs=C.V_OPTIONS, which=None, seed=C.SEED_TRAIN) -> list:
    """Alle Regeln der Tabelle für `par` abstimmen (die Reihenfolge ist C.RULES). `which`: nur diese Namen (Standard alle).
    H = Planungshorizont der Abstimmung; die Zeilen „1 Jahr“ und „5 Jahre“ sind unabhängig davon fest.
    Nur die zwei Refugium-Zeilen verwenden den Refugium-Anteil `q_ref`; alle anderen Regeln laufen ohne Refugium."""
    Y = par.Y
    want = set(which) if which else set(C.RULES)
    pq = with_(par, q=q_ref)
    par = with_(par, q=0.0)
    kw = dict(P=p_train, seed=seed)
    out = {}
    if C.R_NEVER in want:
        b = never()
        out[C.R_NEVER] = _rule(C.R_NEVER, "cal", b)
    if C.R_CAL in want:
        out[C.R_CAL] = _rule(C.R_CAL, "cal", tune(par, "cal", H=H, **kw)[0])
    for name, hh in ((C.R_THR1, 1), (C.R_THR5, 5), (C.R_THRH, H)):
        if name in want:
            b, v, _ = tune_v(par, H=min(hh, Y), vs=vs, **kw)
            out[name] = _rule(name, "thr", b, v=v)
    if C.R_CAP in want:
        b, v, _ = tune_v(par, H=H, extra=dict(cap=[1, 2, 3, 4, 6]), vs=vs, **kw)
        out[C.R_CAP] = _rule(C.R_CAP, "thr", b, v=v)
    if C.R_PAUSE in want:
        out[C.R_PAUSE] = _rule(C.R_PAUSE, "cal", tune(par, "cal", H=H, grid=pause_grid(), **kw)[0])
    for name, mode in ((C.R_CAL_YEAR, "year"), (C.R_CAL_SPRAY, "spray")):
        if name in want:
            out[name] = _rule(name, "cal", tune(par, "cal", mode=mode, H=H, **kw)[0], mode=mode)
    for name, mode in ((C.R_THR_YEAR, "year"), (C.R_THR_SPRAY, "spray")):
        if name in want:
            b, v, _ = tune_v(par, mode=mode, H=H, vs=vs, **kw)
            out[name] = _rule(name, "thr", b, mode=mode, v=v)
    if C.R_SEQ in want:
        b, v, _ = tune_v(par, mode="seq", H=H, extra=dict(Ysw=[1, 2, 3, 4, 6, 8, 12, 20]), grid=dict(S=s_grid(2)), vs=tuple(x for x in vs if x <= 2) or (1,), **kw)
        out[C.R_SEQ] = _rule(C.R_SEQ, "thr", b, mode="seq", v=v)
    if C.R_REF in want:
        b, v, _ = tune_v(pq, H=H, vs=vs, **kw)
        out[C.R_REF] = _rule(C.R_REF, "thr", b, v=v, q=pq.q)
    if C.R_REF_YEAR in want:
        b, v, _ = tune_v(pq, mode="year", H=H, vs=vs, **kw)
        out[C.R_REF_YEAR] = _rule(C.R_REF_YEAR, "thr", b, mode="year", v=v, q=pq.q)
    return [out[n] for n in C.RULES if n in out]


# ----------------------------------------------------------------------------------------------------- Auswertung
def evaluate_rule(par: Par, rule: Rule, paths: int, seed: int, years=None, noise=True, record=False):
    """Eine Regel auf `paths` Auswertungspfaden. Rückgabe: (Summe je Pfad [P], vollständiges Ergebnis der Simulation mit G = 1)."""
    pe = with_(par, q=rule.q)
    res = simulate(pe, rule.kind, rule.arrays(), paths, seed, years=years, mode=rule.mode, noise=noise, v=rule.v, record=record)
    return total(res, pe, years)[0], res


def rule_table(par: Par, rules, paths: int, seed: int) -> list:
    """Kennzahlen je Regel (Liste von dicts in der Reihenfolge von `rules`). Die Zeilen „nie“ und „Kalender“ dienen als Bezug der gepaarten Unterschiede;
    fehlt eine von beiden in `rules`, ist der Unterschied None."""
    ev = {r.name: evaluate_rule(par, r, paths, seed) for r in rules}
    ref_nie = ev[C.R_NEVER][0] if C.R_NEVER in ev else None
    ref_cal = ev[C.R_CAL][0] if C.R_CAL in ev else None
    rows = []
    for r in rules:
        tot, res = ev[r.name]
        m, se = mean_se(tot)
        net = res["net"][0]  # (P, Y)
        row = dict(rule=r.name, total=float(m), se=float(se), params=r.describe(), spec=r.to_json(),
                   sprays=float(res["sprays"][0].mean()), resA20=float(res["resA"][0][:, -1].mean()), resB20=float(res["resB"][0][:, -1].mean()))
        for j, (a, b) in enumerate(((0, 5), (5, 10), (10, 15), (15, 20))):
            row[f"blk{j + 1}"] = float(net[:, a:b].mean())
        for key, ref in (("nie", ref_nie), ("cal", ref_cal)):
            if ref is None:
                row[f"d_{key}"], row[f"d_{key}_se"] = None, None
            else:
                dm, dse = mean_se(tot - ref)
                row[f"d_{key}"], row[f"d_{key}_se"] = float(dm), float(dse)
        rows.append(row)
    return rows


def profile(par: Par, v: int, paths: int, seed: int, n_s=24):
    """Summe über die Jahre in Abhängigkeit von der Schwelle S (n_s Werte, logarithmisch 0,003 bis 1,5) und die Summe von „nie“; für das Profil der Schwellen."""
    S = np.exp(np.linspace(np.log(C.S_GRID_LO), np.log(C.S_GRID_HI), n_s))
    res = simulate(par, "thr", dict(S=S), paths, seed, v=v)
    tot = total(res, par).mean(1)
    nv = total(simulate(par, "cal", never(), paths, seed), par).mean()
    return S, tot, float(nv)
