"""Randfälle und Definitionen (DEMO-PLAYBOOK §12; Lehre aus der Fruchtfolge-Demo): Reihenfolge im Wochenschritt, Abtötung vor dem Wachstum, Jahreswechsel,
Resistenzanteil am Rand, Kontrollwochen, Obergrenze, Pausenjahre, Wechsel; dazu die Reglerränder und ein Zweig-Test: jede Variante weicht von ihrer Nachbarvariante ab."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import sre_constants as C
import sre_model as MD
import sre_rules as R
from sre_model import Par, simulate


def _cal(par, k, w0, **kw):
    return simulate(par, "cal", dict(k=np.array([k]), w0=np.array([w0])), kw.pop("paths", 1), kw.pop("seed", 1), noise=kw.pop("noise", False), **kw)


# ----------------------------------------------------------------------------------------------------------------- Reihenfolge im Wochenschritt
def test_abtoetung_kommt_vor_dem_wachstum():
    """Spritzung in Woche 0: das Wachstum der Woche rechnet mit der Gesamtdichte NACH der Spritzung (1 − N₁/K), für alle vier Genotypen."""
    par = Par(W=3, Y=1)
    res = simulate(par, "cal", dict(k=np.array([1000]), w0=np.array([0])), 1, 1, noise=False, record=True)  # k = 1000, w0 = 0: nur Woche 0 spritzt
    p = 1e-5
    start = [0.01 * x for x in ((1 - p) ** 2, p * (1 - p), (1 - p) * p, p * p)]  # SS, RS, SR, RR
    after = [start[0] * 0.1, start[1] * 0.91, start[2] * 0.1, start[3] * 0.91]  # Wirkstoff A
    N1 = sum(after)
    fits = [1.0, 0.7, 0.7, 0.49]
    for g in range(4):
        assert res["trace_n"][0, 0, 0, g] == pytest.approx(after[g] * math.exp(0.45 * fits[g] * (1 - N1)), rel=1e-12)
    assert res["trace_spray"][0, 0, 0] == 1 and res["trace_spray"][0, 0, 1] == 0


def test_wochenschaden_nutzt_das_mittel_der_dichte_vor_und_nach_dem_wachstum():
    par = Par(W=1, Y=1, m=0.2, p0=0.0)  # eine Woche, nur Anfällige, nie spritzen
    res = _cal(par, 1000, 1000)
    N1 = 0.2
    N2 = N1 * math.exp(0.45 * (1 - N1))
    Nm = 0.5 * (N1 + N2)
    assert res["net"][0, 0, 0] == pytest.approx(100 * (1 - Nm / (Nm + 0.25)), rel=1e-12)
    assert res["net"][0, 0, 0] != pytest.approx(100 * (1 - N2 / (N2 + 0.25)), rel=1e-4)  # nicht die Dichte am Ende allein


def test_jahreswechsel_pool_mit_refugium_in_brüchen():
    """r = 0, ein Jahr mit einer Woche, Spritzung in Woche 0, Refugium q = 1/2: der Pool mischt behandelte Fläche und Refugium je zur Hälfte."""
    par = Par(W=1, Y=1, r=0.0, q=0.5, p0=0.0, kill=0.9, phi=0.9, m=0.01)
    res = simulate(Par(**{**par.__dict__, "p0": 0.001, "pB0": 0.0, "rr_null": True}), "cal", dict(k=np.array([1]), w0=np.array([0])), 1, 1, noise=False)
    p = F(1, 1000)
    S, R_ = F(1, 100) * (1 - p), F(1, 100) * p
    pool_S = F(1, 2) * (S * F(1, 10)) + F(1, 2) * S
    pool_R = F(1, 2) * (R_ * F(91, 100)) + F(1, 2) * R_
    assert res["resA"][0, 0, 0] == pytest.approx(float(pool_R / (pool_S + pool_R)), rel=1e-12)
    assert res["endN"][0, 0, 0] == pytest.approx(float(pool_S + pool_R), rel=1e-12)


def test_naechstes_jahr_beginnt_mit_nu_mal_pool_plus_zuwanderung():
    par = Par(W=1, Y=2, r=0.0, p0=0.0)
    res = _cal(par, 1000, 1000)
    n2 = 0.01 * 0.01 + 0.01  # ν · Pool + m
    assert res["endN"][0, 0, 1] == pytest.approx(n2, rel=1e-13)


# ----------------------------------------------------------------------------------------------------------------- Kalender, Kontrollen, Obergrenze, Pausen
def test_kalender_nie_und_startwoche_am_rand():
    par = Par(Y=2)
    assert _cal(par, 1000, 1000)["sprays"].sum() == 0
    assert _cal(par, 1, 26)["sprays"].sum() == 0  # Start nach der letzten Woche (Wochen 0 bis 25): nie
    assert _cal(par, 1, 25)["sprays"][0, 0].tolist() == [1, 1]  # genau die letzte Woche
    assert _cal(par, 26, 0)["sprays"][0, 0].tolist() == [1, 1]  # k = 26: nur Woche 0
    assert _cal(par, 13, 0)["sprays"][0, 0].tolist() == [2, 2]  # Wochen 0 und 13


def test_kontrolle_beginnt_in_woche_null_und_kostet_je_kontrolle():
    par = Par(Y=1, b=0.0)
    res = simulate(par, "thr", dict(S=np.array([1e-9])), 3, 1, noise=False, v=4, record=True)  # Schwelle unter jeder Dichte: spritzt in jeder Kontrollwoche
    assert np.flatnonzero(res["trace_spray"][0, 0] > 0.5).tolist() == [0, 4, 8, 12, 16, 20, 24]
    assert res["net"][0, 0, 0] == pytest.approx(100 - 7 * par.c_spray - 7 * par.c_scout)


def test_schwelle_ist_streng_groesser():
    """Spritzen heißt gemessen > S: bei genau gleicher Dichte (kein Rauschen) wird nicht gespritzt."""
    par = Par(Y=1, W=1, m=0.01, p0=0.0)
    n0 = 0.01
    assert simulate(par, "thr", dict(S=np.array([n0])), 1, 1, noise=False)["sprays"][0, 0, 0] == 0
    assert simulate(par, "thr", dict(S=np.array([np.nextafter(n0, 0)])), 1, 1, noise=False)["sprays"][0, 0, 0] == 1


def test_obergrenze_zaehlt_je_saison_neu():
    par = Par(Y=3)
    res = simulate(par, "cal", dict(k=np.array([1]), w0=np.array([0]), cap=np.array([2])), 1, 1, noise=False)
    assert res["sprays"][0, 0].tolist() == [2, 2, 2]


def test_pausenjahre_das_erste_jahr_hat_index_null():
    par = Par(Y=7)
    res = simulate(par, "cal", dict(k=np.array([13]), w0=np.array([0]), per=np.array([3])), 1, 1, noise=False)
    assert res["sprays"][0, 0].tolist() == [2, 0, 0, 2, 0, 0, 2]  # Jahre 1, 4, 7
    res = simulate(par, "cal", dict(k=np.array([13]), w0=np.array([0]), per=np.array([1])), 1, 1, noise=False)
    assert res["sprays"][0, 0].tolist() == [2] * 7


# ----------------------------------------------------------------------------------------------------------------- Wirkstoffwahl
def test_wechsel_je_jahr_beginnt_mit_a_und_seq_wechselt_nach_dem_wechseljahr():
    par = Par(Y=4)
    prm = dict(k=np.array([5]), w0=np.array([1]))
    yr = simulate(par, "cal", prm, 1, 1, noise=False, mode="year")
    a = simulate(par, "cal", prm, 1, 1, noise=False, mode="A")
    assert yr["net"][0, 0, 0] == pytest.approx(a["net"][0, 0, 0], rel=1e-12)  # Jahr 1: Wirkstoff A wie ohne Wechsel
    assert yr["resA"][0, 0, 0] > yr["resB"][0, 0, 0]  # A wurde gespritzt: Resistenz gegen A wächst zuerst
    seq = simulate(par, "thr", dict(S=np.array([0.01]), Ysw=np.array([2])), 1, 1, noise=False, mode="seq", v=2)
    one = simulate(par, "thr", dict(S=np.array([0.01])), 1, 1, noise=False, mode="A", v=2, years=2)
    assert np.allclose(seq["net"][0, 0, :2], one["net"][0, 0, :2], atol=1e-12)  # bis einschließlich Jahr 2 Wirkstoff A (Index < Ysw)
    assert seq["resB"][0, 0, 3] > seq["resB"][0, 0, 1]  # ab Jahr 3 wird B gespritzt: Resistenz gegen B wächst


def test_wechsel_je_spritzung_zaehlt_ueber_alle_jahre():
    """Bei drei Spritzungen je Jahr (Wochen 0, 9, 18) spritzt Jahr 1 A, B, A und Jahr 2 B, A, B: nach Wechsel je Jahr ist es A, A, A und B, B, B."""
    par = Par(Y=2)
    prm = dict(k=np.array([9]), w0=np.array([0]))
    sp = simulate(par, "cal", prm, 1, 1, noise=False, mode="spray")
    yr = simulate(par, "cal", prm, 1, 1, noise=False, mode="year")
    assert not np.allclose(sp["net"], yr["net"])
    assert sp["sprays"][0, 0].tolist() == [3, 3] and yr["sprays"][0, 0].tolist() == [3, 3]


# ----------------------------------------------------------------------------------------------------------------- Reglerränder
@pytest.mark.parametrize("kw", [
    dict(phi=1.0, cost_fit=0.0, kill=0.99, m=0.001),
    dict(phi=0.0, cost_fit=0.6, kill=0.5, m=0.1),
    dict(phi=1.0, cost_fit=0.6, c_spray=10.0, c_scout=1.0, sig_meas=1.0),
    dict(cross=1.0, rr_freq=1e-3, q=0.5, disc=0.5),
])
def test_reglerraender_geben_endliche_zahlen(kw):
    par = Par(**kw)
    rules = R.tune_rules(par, 3, 6, q_ref=kw.get("q", 0.2))
    rows = R.rule_table(par, rules, 10, 5)
    for r in rows:
        for key in ("total", "se", "d_nie", "d_cal", "blk1", "blk4", "sprays", "resA20", "resB20"):
            assert math.isfinite(r[key]), (r["rule"], key)
        assert 0.0 <= r["resA20"] <= 1.0 and 0.0 <= r["resB20"] <= 1.0


def test_ein_jahr_und_zwei_pfade():
    res = simulate(Par(Y=1), "cal", dict(k=np.array([5]), w0=np.array([1])), 2, 1)
    assert res["net"].shape == (1, 2, 1)
    m, se = MD.mean_se(res["net"][0, :, 0])
    assert math.isfinite(m) and math.isfinite(se)


def test_zinsfaktor_wirkt_auf_summe_nicht_auf_jahresertrag():
    par, par_d = Par(Y=3), Par(Y=3, disc=0.5)
    a, b = _cal(par, 5, 1), _cal(par_d, 5, 1)
    assert np.allclose(a["net"], b["net"])  # derselbe Jahresertrag
    assert MD.total(b, par_d)[0, 0] == pytest.approx(a["net"][0, 0, 0] + 0.5 * a["net"][0, 0, 1] + 0.25 * a["net"][0, 0, 2])


# ----------------------------------------------------------------------------------------------------------------- Zweig-Test: jede Variante weicht ab
def _sum(par, kind, prm, mode="A", v=1, years=None, paths=10):
    res = simulate(par, kind, {k: np.array([x]) for k, x in prm.items()}, paths, 3, years=years, mode=mode, v=v)
    return MD.total(res, par)[0].mean()


def test_jede_variante_weicht_von_ihrer_nachbarvariante_ab():
    par = Par()
    cal = dict(k=5, w0=1)
    base = _sum(par, "cal", cal)
    assert abs(_sum(par, "cal", cal, mode="year") - base) > 1.0           # Wechsel je Jahr gegen ein Mittel
    assert abs(_sum(par, "cal", dict(k=9, w0=0), mode="spray") - _sum(par, "cal", dict(k=9, w0=0), mode="year")) > 1e-6  # nur wenig: beide Wirkstoffe werden ähnlich oft gebraucht
    thr = dict(S=0.003)
    assert abs(_sum(par, "thr", {**thr, "cap": 2}, v=4) - _sum(par, "thr", thr, v=4)) > 1.0                 # Obergrenze
    pk = dict(k=1, w0=0, cap=3)
    assert abs(_sum(par, "cal", {**pk, "per": 2}) - _sum(par, "cal", {**pk, "per": 1})) > 1.0               # Pausenjahre
    assert abs(_sum(par, "thr", {**thr, "Ysw": 4}, mode="seq", v=2) - _sum(par, "thr", thr, v=2)) > 1.0   # A dann B
    assert abs(_sum(MD.with_(par, q=0.2), "thr", {"S": 0.5}, v=4) - _sum(par, "thr", {"S": 0.5}, v=4)) > 1.0  # Refugium
    assert abs(_sum(MD.with_(par, cross=0.5), "cal", cal, mode="year") - _sum(par, "cal", cal, mode="year")) > 1.0
    assert abs(_sum(MD.with_(par, rr_freq=1e-3), "cal", cal, mode="year") - _sum(par, "cal", cal, mode="year")) > 1.0
    assert abs(_sum(MD.with_(par, sig_meas=1.0), "thr", {"S": 0.05}, v=2) - _sum(par, "thr", {"S": 0.05}, v=2)) > 0.1   # Messrauschen wirkt auf die Schwelle
    assert _sum(MD.with_(par, sig_meas=1.0), "cal", cal) == pytest.approx(base, abs=1e-12)                          # ... nicht auf den Kalender
    assert _sum(MD.with_(par, c_scout=1.0), "cal", cal) == pytest.approx(base, abs=1e-12)                           # Kontrollkosten nur für die Schwelle
    assert abs(_sum(MD.with_(par, c_scout=1.0), "thr", {"S": 0.05}, v=1) - _sum(par, "thr", {"S": 0.05}, v=1)) > 10.0
    assert abs(_sum(MD.with_(par, disc=0.5), "cal", cal) - base) > 100.0


def test_keine_tabellenspalte_besteht_nur_aus_nullen(pre, raster):
    for sc in pre.values():
        for key in ("total", "se", "d_nie", "d_cal", "blk1", "blk2", "blk3", "blk4", "sprays", "resA20", "resB20"):
            assert any(abs(r[key]) > 0 for r in sc["rows"]), (key,)
    import sre_grid as G

    for key in ("S1", "S5", "S20", "sum1", "sum5", "sum20", "nie", "cal", "cap", "pause"):
        assert any(r[key] != 0 for r in raster), key
    assert any(r["resB20"] > 0 for sc in pre.values() for r in sc["rows"] if "Wechsel" in r["rule"])  # der zweite Wirkstoff wird wirklich benutzt
    assert all(p["sprays_B"] > 0 for p in G.load_dp2_politik()) and any(p["dp_sprays"] == 0 for p in G.load_dp_politik())
