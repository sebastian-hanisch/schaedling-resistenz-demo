"""Kern gegen Handrechnung: jede Einheit an einer Mini-Instanz (DEMO-PLAYBOOK §1b); die unabhängigen Rechenwege stehen in test_oracle_schaedling_resistenz.py."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import sre_constants as C
import sre_dp as DP
import sre_grid as G
import sre_hand as H
import sre_model as MD
import sre_rules as R
from sre_model import Par


# ------------------------------------------------------------------------------------------------------- Genotypen
def test_ueberlebenstabelle_von_hand():
    sv = MD.survival(Par())  # kill 0,9, φ 0,9, Kreuzresistenz 0
    assert sv[0] == pytest.approx([0.1, 0.91, 0.1, 0.91]) and sv[1] == pytest.approx([0.1, 0.1, 0.91, 0.91])
    sv = MD.survival(Par(cross=0.5))  # 0,1 + 0,5 · (0,91 − 0,1) = 0,505
    assert sv[0] == pytest.approx([0.1, 0.91, 0.505, 0.91]) and sv[1] == pytest.approx([0.1, 0.505, 0.91, 0.91])
    sv = MD.survival(Par(cross=1.0))
    assert sv[0] == pytest.approx(sv[1])


def test_fitness_je_resistenz():
    assert MD.fitness(Par()) == pytest.approx([1.0, 0.7, 0.7, 0.49])
    assert MD.fitness(Par(cost_fit=0.0)) == pytest.approx([1, 1, 1, 1])


def test_zuwanderungsanteile():
    f = MD.immigrant_freq(Par())  # Produkt der Einzelanteile
    assert f.sum() == pytest.approx(1.0, abs=1e-15) and f[3] == pytest.approx(1e-10) and f[1] == pytest.approx(1e-5 * (1 - 1e-5))
    f = MD.immigrant_freq(Par(rr_freq=1e-4))
    assert f[3] == 1e-4 and f.sum() == pytest.approx(1.0, abs=1e-15) and f[1] == 1e-5
    f = MD.immigrant_freq(Par(rr_null=True, pB0=0.0))
    assert f[2] == 0.0 and f[3] == 0.0 and f[1] == 1e-5


def test_zufallszahlen_mittelwert_eins_und_gleiche_seeds_gleiche_pfade():
    par = Par()
    z, rho, mig = MD.draws(par, 4000, 3, 20)
    assert rho.mean() == pytest.approx(1.0, abs=0.003) and mig.mean() == pytest.approx(1.0, abs=0.01)
    assert z.shape == (4000, 20, 26) and rho.shape == (4000, 20)
    z2, rho2, _ = MD.draws(par, 4000, 3, 20)
    assert np.array_equal(z, z2) and np.array_equal(rho, rho2)
    z0, rho0, mig0 = MD.draws(par, 5, 3, 2, noise=False)
    assert not z0.any() and (rho0 == 1).all() and (mig0 == 1).all()


def test_total_zinst_ab_und_mean_se():
    res = dict(net=np.array([[[10.0, 20.0, 40.0]]]))
    assert MD.total(res, Par(disc=0.5))[0, 0] == pytest.approx(10 + 10 + 10)
    assert MD.total(res, Par(), upto=2)[0, 0] == 30.0
    m, se = MD.mean_se(np.array([1.0, 2.0, 3.0]))
    assert m == 2.0 and se == pytest.approx(1 / math.sqrt(3))


# ------------------------------------------------------------------------------------------------------- Handbeispiel
def test_hand_saison_ohne_spritzen_und_mit_vier_spritzungen():
    rows, net = H.exact_season(set())
    # N = 101/10000: Wochenschaden 25 · N/(N + 1/4) = 25 · 101/2601; vier Wochen
    assert net == 100 - 4 * F(25 * 101, 2601) == F(250000, 2601)
    rows, net = H.exact_season({1, 2, 3, 4})
    S4, R4 = F(1, 100) / 10 ** 4, F(1, 10000) * F(91, 100) ** 4
    assert rows[3]["S"] == S4 and rows[3]["R"] == R4
    loss = sum(F(25) * (F(1, 100) / 10 ** k + F(1, 10000) * F(91, 100) ** k) / (F(1, 100) / 10 ** k + F(1, 10000) * F(91, 100) ** k + F(1, 4)) for k in (1, 2, 3, 4))
    assert net == 100 - loss - 16


def test_hand_ricker_woche_zahlen_der_app():
    wk = H.ricker_week(0.01, 1e-4, True)  # die Zahlen der Tabelle „Von Hand“ Teil 4
    assert wk["S1"] == pytest.approx(0.001) and wk["R1"] == pytest.approx(9.1e-5) and wk["N1"] == pytest.approx(0.001091)
    assert wk["fS"] == pytest.approx(math.exp(0.45 * (1 - 0.001091)), rel=1e-14) and wk["fR"] == pytest.approx(math.exp(0.45 * 0.7 * (1 - 0.001091)), rel=1e-14)
    assert wk["loss"] == pytest.approx(wk["loss"]) and 0 < wk["loss"] < 100 / 26


# ------------------------------------------------------------------------------------------------------- Regeln
def test_gitter_der_abstimmung():
    cg = R.cal_grid()
    assert len(cg["k"]) == 101 and cg["k"][-1] == 1000 and cg["w0"][-1] == 1000  # 76 + 24 Kalender und „nie“ (k = 1 bis 13: 76, dazu 16, 20, 26 mit je 8 Startwochen)
    assert len(R.pause_grid()["k"]) == 3 * 3 * 5 * 5
    sg = R.s_grid()
    assert len(sg) == 36 and sg[0] == pytest.approx(0.003) and sg[-1] == pytest.approx(1.5) and len(R.s_grid(2)) == 18


def test_regel_json_beschreibung_und_nie():
    rule = R.Rule("x", "thr", "year", (("S", 0.0073), ("cap", 2.0)), 4, 0.2)
    assert R.Rule.from_json("x", rule.to_json()) == rule
    assert rule.describe() == "S = 0.0073, höchstens 2 je Saison, Kontrolle alle 4 Wochen, Refugium 20 %"
    assert R.Rule("n", "cal", "A", (("k", 1000.0), ("w0", 1000.0))).describe() == "nie"
    seq = R.Rule("s", "thr", "seq", (("S", 0.6), ("Ysw", 8.0)), 2)
    assert "A in Jahr 1 bis 8, danach B" in seq.describe()
    assert rule.get("cap") == 2.0 and rule.get("per") is None


def test_rule_table_bezug_und_gepaarte_unterschiede():
    par = Par()
    rules = [R._rule(C.R_NEVER, "cal", MD.never()), R._rule(C.R_CAL, "cal", dict(k=np.array([5]), w0=np.array([1])))]
    rows = R.rule_table(par, rules, 20, 7)
    assert rows[0]["d_nie"] == 0.0 and rows[1]["d_cal"] == 0.0 and rows[1]["d_cal_se"] == 0.0
    assert rows[1]["d_nie"] == pytest.approx(rows[1]["total"] - rows[0]["total"], rel=1e-12)
    assert rows[1]["sprays"] == 5.0  # Wochen 1, 6, 11, 16, 21 in jedem Jahr
    only_cal = R.rule_table(par, rules[1:], 20, 7)
    assert only_cal[0]["d_nie"] is None and only_cal[0]["d_cal"] == 0.0  # ohne „nie“ fehlt der Bezug; der Kalender ist sein eigener


def test_schnellabstimmung_ohne_refugium_gleicht_den_nicht_refugium_zeilen():
    par = Par(Y=3)
    rules = R.tune_rules(par, 3, 8, q_ref=0.0, which=[C.R_THRH, C.R_REF])
    by = {r.name: r for r in rules}
    assert by[C.R_REF].prm == by[C.R_THRH].prm and by[C.R_REF].v == by[C.R_THRH].v and by[C.R_REF].q == 0.0


def test_planungshorizont_wirkt_nur_auf_die_horizont_zeilen():
    par = Par(Y=4)
    a = {r.name: r for r in R.tune_rules(par, 4, 8, which=[C.R_THR1, C.R_THR5, C.R_THRH, C.R_CAL])}
    b = {r.name: r for r in R.tune_rules(par, 1, 8, which=[C.R_THR1, C.R_THR5, C.R_THRH, C.R_CAL])}
    assert a[C.R_THR1] == b[C.R_THR1] and a[C.R_THR5] == b[C.R_THR5]  # fest 1 und 5 Jahre
    assert b[C.R_THRH].prm == b[C.R_THR1].prm and b[C.R_THRH].v == b[C.R_THR1].v  # Horizont 1 = die Kurzsicht-Zeile


def test_profil_hat_24_schwellen_und_nie_als_bezug():
    S, tot, nie = R.profile(Par(Y=3), 2, 10, 4, n_s=24)
    assert len(S) == 24 and len(tot) == 24 and nie > 0 and S[0] == pytest.approx(0.003)


# ------------------------------------------------------------------------------------------------------- Rückwärtsrechnung
def test_gitter_und_interpolation_linearer_funktionen_exakt():
    grid = DP.Grid(-3.0, 0.0, 0.5)
    assert len(grid.axis) == 7
    ax = grid.axis
    V = np.add.outer(2.0 * ax, -3.0 * ax)  # 2x − 3y ist linear: bilineare Interpolation trifft sie exakt
    S, R_ = np.array([10 ** -1.2, 10 ** -2.9]), np.array([10 ** -0.3, 10 ** -1.1])
    expected = 2 * np.log10(S) - 3 * np.log10(R_)
    assert DP.interp(V, S, R_, grid) == pytest.approx(expected, abs=1e-12)
    assert DP.interp(V, np.array([1e-30]), np.array([1.0]), grid) == pytest.approx(V[0, -1])  # außerhalb: nächstes Randfeld


def test_winter_und_startzustand():
    par = Par()
    assert DP.start_state(par) == (0.01 * (1 - 1e-5), 0.01 * 1e-5)
    assert DP.winter(par, 0.5, 0.2) == (0.01 * 0.5 + 0.01 * (1 - 1e-5), 0.01 * 0.2 + 0.01 * 1e-5)
    p2 = DP.det_par(Par(q=0.2, c_scout=0.1, disc=0.8))
    assert (p2.q, p2.c_scout, p2.disc, p2.pB0, p2.rr_null) == (0.0, 0.0, 1.0, 0.0, True)
    p3 = DP.det_par2(Par())
    assert p3.pB0 == 1e-5 and p3.rr_null and p3.q == 0.0


def test_myopische_rueckwaertsrechnung_beginnt_jedes_jahr_bei_null():
    """Mit myopic=True zählt nur das laufende Jahr: der Wert zu Beginn ist genau der Wert einer Rechnung über ein einziges Jahr."""
    kw = dict(W=8, c_spray=1.0, p0=1e-3)
    par2, par1 = DP.det_par(Par(Y=2, **kw)), DP.det_par(Par(Y=1, **kw))
    V_myo, _ = DP.solve(par2, myopic=True)
    V_one, _ = DP.solve(par1)
    V_far, _ = DP.solve(par2)
    raw = lambda V, par: DP.value_at(V, par) - 100.0 * par.Y
    assert raw(V_myo, par2) == pytest.approx(raw(V_one, par1), abs=1e-12)
    assert raw(V_far, par2) < raw(V_myo, par2)  # zwei Jahre kosten mehr Verlust als eines: der Wert (ohne den Grundwert) ist kleiner


# ------------------------------------------------------------------------------------------------------- Tabellen (CSV)
def test_csv_tabellen_sind_vollstaendig(pre, raster):
    assert len(pre) == 6 and all(len(sc["rows"]) == 14 for sc in pre.values())
    assert len(raster) == len(C.GRID_PHI) * len(C.GRID_COST) == 42
    assert G.raster_lookup(raster, 0.9, 0.3)["S20"] == pytest.approx(0.881, abs=0.001)  # Kontrollabstand 2 Wochen (die Tabelle des Grundfalls wählt 4: S = 0,737)
    assert G.raster_lookup(raster, 0.91, 0.3) is None
    hz = G.load_horizont()
    assert [int(r["H"]) for r in hz] == list(C.HORIZONS)
    dpw = G.load_dp_werte()
    assert list(dpw) == list(C.PRESETS) and dpw[next(iter(dpw))]["dp2"] is not None
    assert len(G.load_dp_politik()) == 20 and len(G.load_dp2_politik()) == 20
