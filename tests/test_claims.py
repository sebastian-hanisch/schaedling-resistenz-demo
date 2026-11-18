"""Die Zahlen der README („Befunde“) werden hier geprüft. Die vorgerechneten Tabellen (data/*.csv) werden gelesen und als Zählungen und Bänder verglichen; die Regeln der
Grundfall-Tabelle werden auf den 2000 Auswertungspfaden NEU gerechnet (Band: Plattformen unterscheiden sich in der letzten Binärstelle, DEMO-PLAYBOOK §5), ebenso das Schwellenprofil,
das Messrauschen, die Doppelresistenz und eine Rasterzelle. Nicht jede Zahl der README steht hier als Zahl: die Zahlen der Rückwärtsrechnung, der Politik und der anderen Presets sind
Bänder auf die CSV, nicht auf eine unabhängige Neu-Rechnung; Laufzeiten stehen nur im Text. Die Abstimmung selbst (das Auffinden der Parameter) wird nur an einer Rasterzelle neu gerechnet."""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import sre_constants as C
import sre_evaluation as ev
import sre_grid as G
import sre_rules as R
from sre_model import simulate, total
from sre_scenario import DEFAULTS, par_from_settings, settings_of

ROOT = Path(__file__).resolve().parent.parent
GRUND = "Grundfall (φ 0,9, c 0,3)"


@pytest.fixture(scope="module")
def grund(pre):
    return {r["rule"]: r for r in pre[GRUND]["rows"]}


def _rows(pre, name):
    return {r["rule"]: r for r in pre[name]["rows"]}


# ------------------------------------------------------------------------------------------------------------ Regeltabelle Grundfall
def test_regeltabelle_grundfall_summen(grund):
    expected = {C.R_NEVER: 728, C.R_CAL: 979, C.R_THR1: 349, C.R_THR5: 683, C.R_THRH: 811, C.R_CAP: 1012, C.R_PAUSE: 1082, C.R_CAL_YEAR: 1533, C.R_CAL_SPRAY: 1533,
                C.R_THR_YEAR: 1512, C.R_THR_SPRAY: 1513, C.R_SEQ: 816, C.R_REF: 802, C.R_REF_YEAR: 1351}
    assert set(expected) == set(C.RULES)
    for name, value in expected.items():
        assert grund[name]["total"] == pytest.approx(value, abs=0.6), name
        assert grund[name]["se"] < 1.5  # Standardfehler 0,3 bis 1,4
    assert min(r["se"] for r in grund.values()) >= 0.25


def test_regeltabelle_grundfall_parameter_und_spritzungen(grund):
    spec = {n: R.Rule.from_json(n, r["spec"]) for n, r in grund.items()}
    assert (spec[C.R_CAL].get("k"), spec[C.R_CAL].get("w0")) == (20, 4) and grund[C.R_CAL]["sprays"] == pytest.approx(2.0, abs=0.05)
    assert spec[C.R_THR1].get("S") == pytest.approx(0.0073, abs=0.0001) and spec[C.R_THR1].v == 2 and grund[C.R_THR1]["sprays"] == pytest.approx(11.8, abs=0.05)
    assert spec[C.R_THRH].get("S") == pytest.approx(0.74, abs=0.005) and spec[C.R_THRH].v == 4 and grund[C.R_THRH]["sprays"] == pytest.approx(1.8, abs=0.05)
    assert spec[C.R_CAP].get("cap") == 2
    p = spec[C.R_PAUSE]
    assert (p.get("k"), p.get("w0"), p.get("cap"), p.get("per")) == (1, 0, 3, 2)
    assert spec[C.R_SEQ].get("Ysw") == 8
    assert grund[C.R_CAL_YEAR]["resB20"] > 0 and grund[C.R_CAL_SPRAY]["resB20"] > 0


def test_einschwingzeit_bloecke(grund):
    blocks = lambda n: [grund[n][f"blk{j}"] for j in (1, 2, 3, 4)]
    assert blocks(C.R_NEVER) == pytest.approx([37.1, 36.2, 36.2, 36.2], abs=0.06)
    assert blocks(C.R_THR1) == pytest.approx([52.1, 6.3, 5.7, 5.6], abs=0.06)
    assert blocks(C.R_CAL) == pytest.approx([49.8, 49.5, 49.3, 47.2], abs=0.06)
    assert blocks(C.R_THRH) == pytest.approx([41.7, 41.3, 40.5, 38.8], abs=0.06)


def test_kurzsicht_abstaende(grund):
    assert grund[C.R_THR1]["d_nie"] == pytest.approx(-380, abs=1.0) and grund[C.R_THRH]["d_nie"] == pytest.approx(83, abs=1.0)
    assert round(grund[C.R_THRH]["total"]) - round(grund[C.R_THR1]["total"]) == 462
    assert grund[C.R_THR1]["total"] < grund[C.R_NEVER]["total"] < grund[C.R_THRH]["total"]
    assert grund[C.R_THR1]["resA20"] > 0.99  # die Resistenz hat sich durchgesetzt


def test_kurzsicht_schwelle_spritzt_ab_jahr_6_rund_13_mal(grund):
    """README: ab Jahr 6 spritzt die Kurzsicht-Schwelle rund 13-mal je Jahr (Band 12,5 bis 13,5), im Mittel aller 20 Jahre 11,8 (Jahre 1 bis 5: rund 8)."""
    _, res = R.evaluate_rule(par_from_settings(DEFAULTS), R.Rule.from_json("x", grund[C.R_THR1]["spec"]), C.P_EVAL, C.SEED_EVAL)
    je_jahr = res["sprays"][0].mean(0)
    assert 12.5 < je_jahr[5:].min() and je_jahr[5:].max() < 13.5
    assert je_jahr.mean() == pytest.approx(11.8, abs=0.05) and je_jahr[:5].mean() == pytest.approx(8.4, abs=0.2)


def test_pausen_wechsel_abstaende(grund):
    assert grund[C.R_PAUSE]["d_cal"] == pytest.approx(102.5, abs=1.0) and grund[C.R_PAUSE]["d_cal_se"] == pytest.approx(0.35, abs=0.1)
    assert 100 * grund[C.R_PAUSE]["d_cal"] / grund[C.R_CAL]["total"] == pytest.approx(10.5, abs=0.1)
    assert grund[C.R_CAL_YEAR]["d_cal"] == pytest.approx(554, abs=1.5)
    assert grund[C.R_CAL_YEAR]["total"] - grund[C.R_PAUSE]["total"] == pytest.approx(451, abs=1.5)
    assert grund[C.R_REF]["total"] - grund[C.R_THRH]["total"] == pytest.approx(-9, abs=1.5) and grund[C.R_REF_YEAR]["total"] - grund[C.R_THR_YEAR]["total"] == pytest.approx(-161, abs=1.5)


def test_standardfehler_im_grundfall_und_in_den_presets(grund, pre):
    """README: Standardfehler der Summen im Grundfall 0,3 bis 1,4 Punkte, in den Presets bis 2,9."""
    ses = [r["se"] for r in grund.values()]
    assert (round(min(ses), 1), round(max(ses), 1)) == (0.3, 1.4)
    assert round(max(r["se"] for sc in pre.values() for r in sc["rows"]), 1) == 2.9


def test_kontrollabstaende_der_abstimmung():
    """README Methodik: 1, 2 oder 4 Wochen in den Regeltabellen, 2 im Raster, 2 oder 4 in den Horizonten."""
    assert C.V_OPTIONS == (1, 2, 4) and C.V_SWEEP == (2, 4)
    assert {int(r["v"]) for r in G.load_horizont()} <= {2, 4}


def test_kreuzresistenz_wirkt_auch_auf_regeln_mit_einem_wirkstoff(grund):
    """README und Hilfetext: Regeln des Grundfalls (nicht neu abgestimmt) fallen bei Kreuzresistenz 1 um 6 (Kalender) bzw. 16 (Kurzsicht-Schwelle) Punkte, bei 0,5 praktisch nichts."""
    def tot(cross, name):
        return R.evaluate_rule(par_from_settings({**DEFAULTS, "cross": cross}), R.Rule.from_json(name, grund[name]["spec"]), C.P_EVAL, C.SEED_EVAL)[0].mean()

    for name, fall in ((C.R_CAL, 6.0), (C.R_THR1, 16.0)):
        base = tot(0.0, name)
        assert base - tot(1.0, name) == pytest.approx(fall, abs=0.5), name
        assert abs(base - tot(0.5, name)) < 0.1, name
    assert "um 6 bis 16 Punkte" in (ROOT / "app.py").read_text(encoding="utf-8")

@pytest.mark.parametrize("name", list(C.RULES))
def test_regeln_des_grundfalls_neu_gerechnet(name, grund):
    """Jede Zeile der Tabelle: die abgestimmte Regel auf den 2000 Auswertungspfaden (Seed 22) neu gerechnet, Band aus Standardfehlern und einem Punkt."""
    par = par_from_settings(DEFAULTS)
    tot, _ = R.evaluate_rule(par, R.Rule.from_json(name, grund[name]["spec"]), C.P_EVAL, C.SEED_EVAL)
    assert tot.mean() == pytest.approx(grund[name]["total"], abs=max(1.5, 4 * grund[name]["se"]))


# ------------------------------------------------------------------------------------------------------------ Presets
def test_presets_zahlen(pre):
    teuer, schwach, billig, zins, kreuz = (_rows(pre, n) for n in ("Resistenz teuer (c 0,5)", "Resistenz schwach (φ 0,5)", "Resistenz billig und stark (φ 1,0, c 0,1)",
                                                                   "Kurzer Horizont (Zins 0,8)", "Kreuzresistenz 0,5"))
    assert (teuer[C.R_THR1]["total"], teuer[C.R_THRH]["total"]) == pytest.approx((1253, 1324), abs=1.0)
    assert round(teuer[C.R_THRH]["total"]) - round(teuer[C.R_THR1]["total"]) == 71  # Hilfetext und README: Abstand 71 in der Regeltabelle
    rz = next(r for r in G.load_raster() if (r["phi"], r["cost"]) == (0.9, 0.5))
    assert (round(rz["sum1"]), round(rz["sum20"])) == (1250, 1256) and round(rz["sum20"]) - round(rz["sum1"]) == 6  # ... und 6 im Raster
    assert "6 Punkte auseinander" in C.PRESET_HELP["Resistenz teuer (c 0,5)"] and "Kontrollabstand 71" in C.PRESET_HELP["Resistenz teuer (c 0,5)"]
    assert (schwach[C.R_THR1]["total"], schwach[C.R_THRH]["total"], schwach[C.R_CAL]["total"]) == pytest.approx((1489, 1496, 1475), abs=1.0)
    assert (billig[C.R_THR1]["total"], billig[C.R_NEVER]["total"], billig[C.R_THRH]["total"], billig[C.R_CAL]["total"]) == pytest.approx((392, 728, 747, 876), abs=1.0)
    rot = max(billig[n]["total"] for n in (C.R_CAL_YEAR, C.R_CAL_SPRAY, C.R_THR_YEAR, C.R_THR_SPRAY))
    assert rot == pytest.approx(961, abs=1.0) and billig[C.R_PAUSE]["total"] == pytest.approx(899, abs=1.0) and rot - billig[C.R_PAUSE]["total"] == pytest.approx(62, abs=1.5)
    assert (zins[C.R_NEVER]["total"], zins[C.R_THR1]["total"], zins[C.R_THR5]["total"], zins[C.R_THRH]["total"]) == pytest.approx((183, 207, 257, 257), abs=1.0)
    assert R.Rule.from_json("x", zins[C.R_THRH]["spec"]).get("S") == pytest.approx(0.0051, abs=0.0001)
    assert zins[C.R_THR1]["total"] > zins[C.R_NEVER]["total"] and R.Rule.from_json("x", zins[C.R_THRH]["spec"]).get("S") < 0.1  # kein „fast nie spritzen“
    assert round(zins[C.R_THRH]["total"]) - round(zins[C.R_THR1]["total"]) == 50 and "50 abgezinste Punkte" in C.PRESET_HELP["Kurzer Horizont (Zins 0,8)"]  # Kurzsicht kostet weiter
    assert ev.kurzsicht_verdikt(zins[C.R_THR1]["total"], zins[C.R_NEVER]["total"], zins[C.R_THRH]["total"])[1].startswith("**Kurzsicht kostet spürbar")
    rot_k = max(kreuz[n]["total"] for n in (C.R_CAL_YEAR, C.R_CAL_SPRAY, C.R_THR_YEAR, C.R_THR_SPRAY))
    single = max(kreuz[n]["total"] for n in (C.R_CAL, C.R_THR1, C.R_THR5, C.R_THRH, C.R_CAP, C.R_PAUSE))
    assert (rot_k, single, kreuz[C.R_CAL_YEAR]["total"]) == pytest.approx((1270, 1082, 1245), abs=1.0)
    s = _rows(pre, "Resistenz schwach (φ 0,5)")[C.R_PAUSE]
    assert s["d_cal"] == pytest.approx(24, abs=1.0)
    assert s["d_cal"] > 3 * s["d_cal_se"]


def test_refugium_ist_in_allen_sechs_szenarien_schlechter(pre):
    assert len(pre) == 6
    worse1 = sum(_rows(pre, n)[C.R_REF]["total"] < _rows(pre, n)[C.R_THRH]["total"] for n in pre)
    worse2 = sum(_rows(pre, n)[C.R_REF_YEAR]["total"] < _rows(pre, n)[C.R_THR_YEAR]["total"] for n in pre)
    assert (worse1, worse2) == (6, 6)


# ------------------------------------------------------------------------------------------------------------ Raster
def test_raster_zaehlungen(raster):
    assert len(raster) == 42
    dil = {(r["phi"], r["cost"]) for r in raster if r["S20"] >= C.KIPP_S}
    assert len(dil) == 19
    bound = {phi: max((c for p, c in dil if p == phi), default=None) for phi in C.GRID_PHI}
    assert bound == {0.5: None, 0.6: 0.0, 0.7: 0.2, 0.8: 0.3, 0.9: 0.4, 1.0: 0.5}
    for phi in C.GRID_PHI:  # unter der Grenze liegen alle kleineren c ebenfalls im Dilemma (keine Lücken)
        cs = sorted(c for p, c in dil if p == phi)
        assert cs == [c for c in C.GRID_COST if bound[phi] is not None and c <= bound[phi]]
    assert sum(r["sum1"] < r["nie"] for r in raster) == 17


def test_raster_sprung_bei_c_03_und_gleiche_schwellen_bei_c_06(raster):
    s20 = {r["phi"]: r["S20"] for r in raster if r["cost"] == 0.3}
    assert s20[0.7] == pytest.approx(0.003, abs=0.0005) and s20[0.8] == pytest.approx(0.881, abs=0.001)
    for r in raster:
        if r["cost"] == 0.6 and r["phi"] <= 0.9:
            assert all(0.0055 < r[k] < 0.0075 for k in ("S1", "S5", "S20"))
            assert round(r["S1"], 3) == 0.006 and round(r["S20"], 3) == 0.007  # benachbarte Gitterwerte, nicht gleich
            assert abs(r["sum20"] - r["sum1"]) < 1.0  # gleichwertig: die Summen unterscheiden sich um weniger als einen Punkt


def test_raster_pausen_und_schwelle_gegen_kalender(raster):
    assert sum(r["pause"] > r["cal"] for r in raster) == 37
    best = max(raster, key=lambda r: r["pause"] - r["cal"])
    assert (best["phi"], best["cost"]) == (0.8, 0.3) and best["pause"] - best["cal"] == pytest.approx(126.4, abs=0.5)
    assert {(r["phi"], r["cost"]) for r in raster if r["pause"] < r["cal"]} == {(0.5, 0.0), (0.5, 0.1), (0.5, 0.2), (0.6, 0.2), (0.6, 0.3)}
    sub = [r for r in raster if r["phi"] >= 0.7]
    assert len(sub) == 28 and sum(r["sum20"] < r["pause"] for r in sub) == 27 and sum(r["sum20"] <= r["cal"] for r in sub) == 26
    ausnahme = [r for r in sub if r["sum20"] > r["cal"]]
    assert {(r["phi"], r["cost"]) for r in ausnahme} == {(0.7, 0.3), (0.7, 0.4)}
    assert max(r["sum20"] - r["cal"] for r in ausnahme) == pytest.approx(12, abs=1.5)
    assert min(r["sum20"] - r["cal"] for r in ausnahme) < 1.0  # (0,7; 0,3): gleich bis auf Bruchteile eines Punkts


def test_raster_zelle_neu_abgestimmt():
    """Eine Zelle (φ 0,8, c 0,3) wird mit allen drei Horizonten neu abgestimmt: die Schwelle auf 20 Jahre liegt weit über der auf 1 Jahr (Band, nicht Gitterpunkt)."""
    import sre_grid as g

    cell = g.raster_cell((0.8, 0.3))
    ref = g.raster_lookup(G.load_raster(), 0.8, 0.3)
    assert cell["S1"] < 0.02 and cell["S20"] >= C.KIPP_S
    assert cell["sum1"] == pytest.approx(ref["sum1"], abs=4.0) and cell["sum20"] == pytest.approx(ref["sum20"], abs=4.0) and cell["nie"] == pytest.approx(728, abs=1.0)


# ------------------------------------------------------------------------------------------------------------ Horizonte, Profil
def test_horizonte():
    hz = {int(r["H"]): r for r in G.load_horizont()}
    assert [round(hz[h]["sum_thr"]) for h in (1, 2, 3, 5, 10, 20)] == pytest.approx([348, 684, 684, 684, 699, 811], abs=1)
    assert [round(hz[h]["S"], 3) for h in (1, 2, 3, 5, 10)] == pytest.approx([0.006, 0.004, 0.006, 0.006, 0.021], abs=0.0005) and hz[20]["S"] == pytest.approx(0.74, abs=0.005)
    assert [round(hz[h]["sum_cal"]) for h in (1, 5, 10, 20)] == pytest.approx([806, 859, 960, 979], abs=1) and [int(hz[h]["k"]) for h in (1, 5, 10, 20)] == [5, 7, 13, 20]
    kurz = [hz[h]["S"] for h in (1, 2, 3, 5, 10)]  # App-Bildunterschrift: kurze Horizonte (1 bis 10 Jahre) wählen 0,004 bis 0,021, erst H = 20 eine große Schwelle (0,74)
    assert (round(min(kurz), 3), round(max(kurz), 3)) == (0.004, 0.021) and hz[20]["S"] > 30 * max(kurz)
    assert "Kurze Horizonte (1 bis 10 Jahre) wählen kleine Schwellen (0,004 bis 0,021), erst die Abstimmung auf 20 Jahre eine große (0,74)" in (ROOT / "app.py").read_text(encoding="utf-8")
    assert hz[2]["S"] < hz[1]["S"]  # nicht monoton im Horizont: H = 2 liegt unter H = 1


def test_schwellenprofil_grundfall_neu_gerechnet():
    S, tot, nie = R.profile(par_from_settings(DEFAULTS), 2, 1000, 22)
    i = int(np.argmax(tot))
    assert S[i] == pytest.approx(0.87, abs=0.01) and tot[i] == pytest.approx(796, abs=3)
    assert np.all(np.diff(tot[: i + 1]) > 0)  # steigt bis zur besten Schwelle
    assert tot[22] == pytest.approx(770, abs=3) and S[22] == pytest.approx(1.145, abs=0.01) and tot[23] == pytest.approx(727, abs=2) and nie == pytest.approx(728, abs=1)
    assert max(tot[S <= 0.51]) < 650
    assert i < 22 and tot[i] > tot[22] > tot[23]  # inneres Maximum, nicht am Rand des Gitters


def test_schwellenprofil_spritzzahlen_der_besten_und_der_obersten_schwelle():
    """README: die beste Schwelle (S = 0,87) spritzt rund 1,8-mal je Jahr; bei S = 1,5 sind es noch 0,4 Spritzungen je Jahr (Messrauschen löst aus)."""
    par = par_from_settings(DEFAULTS)
    S = np.exp(np.linspace(np.log(C.S_GRID_LO), np.log(C.S_GRID_HI), 24))
    res = simulate(par, "thr", dict(S=S), 1000, 22, v=2)
    i = int(np.argmax(total(res, par).mean(1)))
    assert S[i] == pytest.approx(0.87, abs=0.01)
    assert res["sprays"][i].mean() == pytest.approx(1.8, abs=0.1)
    assert res["sprays"][23].mean() == pytest.approx(0.4, abs=0.05) and S[23] == pytest.approx(1.5)


# ------------------------------------------------------------------------------------------------------------ Rückwärtsrechnung
def test_rueckwaertsrechnung_grundfall_werte():
    dpw = G.load_dp_werte()[GRUND]
    assert dpw["dp"] == pytest.approx(1123.5, abs=0.1) and dpw["dp_forward"] == pytest.approx(1123.4, abs=0.1) and abs(dpw["dp"] - dpw["dp_forward"]) < 0.2  # Gitterfehler
    assert dpw["myopic"] == pytest.approx(1009.9, abs=0.1) and 100 * (dpw["dp"] / dpw["myopic"] - 1) == pytest.approx(11.3, abs=0.05)
    assert dpw["cal"] == pytest.approx(983.3, abs=0.1) and dpw["cal_k"] == 24 and dpw["thr"] == pytest.approx(816.1, abs=0.1) and dpw["thr_S"] == pytest.approx(0.91, abs=0.005)
    assert dpw["nie"] == pytest.approx(716.7, abs=0.1)
    assert dpw["dp2"] == pytest.approx(1607.1, abs=0.1) and dpw["dp2_forward"] == pytest.approx(1607.6, abs=0.1)
    assert 100 * 1533 / dpw["dp2_forward"] == pytest.approx(95, abs=0.5)  # Anhalt, andere Instanz


def test_rueckwaertsrechnung_politik_grundfall():
    pol = G.load_dp_politik()
    assert [p["dp_sprays"] for p in pol] == [4, 4, 0, 0, 5, 0, 0, 5, 0, 0, 5, 0, 0, 5, 1, 0, 0, 5, 4, 3]
    assert pol[0]["dp_weeks"] == [0, 1, 2, 3] and pol[1]["dp_weeks"] == [0, 1, 2, 3]
    assert [pol[j]["dp_weeks"] for j in (4, 7)] == [[0, 1, 2, 3, 5]] * 2  # Jahre 5 und 8: Wochen 0 bis 3 und 5
    assert [pol[j]["dp_weeks"] for j in (10, 13, 17)] == [[0, 1, 2, 3, 4]] * 3  # Jahre 11, 14 und 18: Wochen 0 bis 4
    assert pol[4]["dp_res"] == pytest.approx(0.165, abs=0.001) and pol[5]["dp_res"] == pytest.approx(0.0012, abs=0.0001)
    assert [p["my_sprays"] for p in pol][:7] == [4, 4, 4, 3, 2, 2, 1] and max(p["my_sprays"] for p in pol) == 4 and min(p["my_sprays"] for p in pol) == 1
    late = [p["my_res"] for p in pol[2:]]
    assert min(late) == pytest.approx(0.52, abs=0.005) and max(late) == pytest.approx(0.89, abs=0.005)


def test_zwei_wirkstoffe_jedes_jahr_zweimal_a_und_b():
    pol = G.load_dp2_politik()
    assert len(pol) == 20 and all(p["sprays_A"] == 2 and p["sprays_B"] == 2 for p in pol)
    assert max(max(p["res_A"], p["res_B"]) for p in pol) < 1e-4


# ------------------------------------------------------------------------------------------------------------ Messrauschen, Doppelresistenz (neu gerechnet)
def _total(change, name, grund):
    par = par_from_settings({**DEFAULTS, **change})
    return R.evaluate_rule(par, R.Rule.from_json(name, grund[name]["spec"]), C.P_EVAL, C.SEED_EVAL)[0].mean()


def test_messrauschen_ist_kein_hebel(grund):
    sig = (0.0, 0.3, 0.6, 1.0)
    thrh = [_total({"meas": s}, C.R_THRH, grund) for s in sig]
    thr1 = [_total({"meas": s}, C.R_THR1, grund) for s in sig]
    cap = [_total({"meas": s}, C.R_CAP, grund) for s in sig]
    assert thrh == pytest.approx([792, 811, 821, 829], abs=2.5) and thr1 == pytest.approx([346, 349, 358, 374], abs=2.5) and cap == pytest.approx([1012, 1012, 1012, 1009], abs=2.5)
    assert max(max(x) - min(x) for x in (thrh, thr1, cap)) == pytest.approx(37, abs=3)
    for s in (0.0, 1.0):  # Kalender und Pausenkalender messen nicht
        assert _total({"meas": s}, C.R_CAL, grund) == pytest.approx(979, abs=0.6) and _total({"meas": s}, C.R_PAUSE, grund) == pytest.approx(1082, abs=1.0)


def test_doppelresistenz_wechsel_bleibt_ueber_dem_pausenkalender(grund):
    vals = [_total({"rr": e}, C.R_CAL_YEAR, grund) for e in (-10, -8, -6, -4, -3)]
    assert vals == pytest.approx([1533, 1524, 1472, 1351, 1274], abs=2.5)
    assert all(v > grund[C.R_PAUSE]["total"] for v in vals) and all(a > b for a, b in zip(vals, vals[1:]))


# ------------------------------------------------------------------------------------------------------------ Handbeispiel, README
def test_handbeispiel_zahlen():
    from fractions import Fraction as F

    import sre_hand as H

    assert H.share_start() == F(1, 101) and H.share_after(H.share_start(), 1, F(1, 10), F(91, 100)) == F(91, 1091) and float(F(91, 1091)) == pytest.approx(0.0834, abs=0.0001)
    assert H.exact_season(set())[1] == F(250000, 2601) and float(F(250000, 2601)) == pytest.approx(96.1, abs=0.05)


def test_modelldetails_der_readme_stehen_im_code():
    """Anfangswerte, Zufall, Regeln im Einzelnen, Rückwärtsrechnung und Reglerbereiche des README-Abschnitts „Ablauf, Zufall und Regeln im Einzelnen“ stimmen mit dem Code überein."""
    import inspect

    import sre_dp as DP
    from sre_model import Par, draws, immigrant_freq, never, survival

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for text in ("π_SS = (1 − p)², π_RS = π_SR = p (1 − p), π_RR = p²", "(Kosten 4 · (1 − q))", "σ_w = 0,15, σ_m = 0,5", "Median 1, ohne Ausgleich des Mittelwerts", "101 Kalender", "225 Kombinationen",
                 "{1, 2, 3, 4, 6, 8, 12, 20}", "jeder zweite Schwellenwert (18 Werte)", "von −9 bis 0,7 in Schritten von 0,05", "von −8 bis 0,6 in Schritten von 0,1", "Der Wert nach Woche 520 ist 0",
                 "Pfadzahl 50 bis 1000 (200)", "Zufalls-Seed 0 bis 9999 (22)", "Refugium-Anteil q 0 bis 0,5 (0,2)", "Zinsfaktor 0,5 bis 1 (1)", "Planungshorizont 1 bis 20 Jahre (20)"):
        assert text in readme, text
    # Reglerbereiche und Standardwerte
    assert [(C.PHI_MIN, C.PHI_MAX, C.DEFAULT_PHI), (C.COST_MIN, C.COST_MAX, C.DEFAULT_COST), (C.KILL_MIN, C.KILL_MAX, C.DEFAULT_KILL), (C.CROSS_MIN, C.CROSS_MAX, C.DEFAULT_CROSS),
            (C.RR_MIN, C.RR_MAX, C.DEFAULT_RR), (C.M_MIN, C.M_MAX, C.DEFAULT_M), (C.SPRAY_MIN, C.SPRAY_MAX, C.DEFAULT_SPRAY), (C.SCOUT_MIN, C.SCOUT_MAX, C.DEFAULT_SCOUT),
            (C.MEAS_MIN, C.MEAS_MAX, C.DEFAULT_MEAS), (C.Q_MIN, C.Q_MAX, C.DEFAULT_Q), (C.HORIZON_MIN, C.HORIZON_MAX, C.DEFAULT_HORIZON), (C.DISC_MIN, C.DISC_MAX, C.DEFAULT_DISC),
            (C.SEED_MIN, C.SEED_MAX, C.DEFAULT_SEED), (C.PATHS_MIN, C.PATHS_MAX, C.DEFAULT_PATHS)] == [
        (0.0, 1.0, 0.9), (0.0, 0.6, 0.3), (0.5, 0.99, 0.9), (0.0, 1.0, 0.0), (-10, -3, -10), (0.001, 0.1, 0.01), (1.0, 10.0, 4.0), (0.0, 1.0, 0.1), (0.0, 1.0, 0.3), (0.0, 0.5, 0.2),
        (1, 20, 20), (0.5, 1.0, 1.0), (0, 9999, 22), (50, 1000, 200)]
    par = Par()
    assert (par.W, par.r, par.K, par.h, par.nu, par.m, par.p0, par.sig_w, par.sig_mig, par.sig_meas, par.c_spray, par.c_scout) == (26, 0.45, 1.0, 0.25, 0.01, 0.01, 1e-5, 0.15, 0.5, 0.3, 4.0, 0.1)
    # Anfangswerte: π aus p, mit vorgegebenem π_RR wird π_SS verkleinert
    p = 1e-5
    assert immigrant_freq(par) == pytest.approx([(1 - p) ** 2, p * (1 - p), p * (1 - p), p * p], rel=1e-12)
    assert immigrant_freq(Par(rr_freq=1e-4)) == pytest.approx([1 - 2 * p - 1e-4, p, p, 1e-4], rel=1e-12)
    # Zufall: Wetter und Zuwanderung je Pfad und Jahr (Mittelwert 1), Messfehler je Woche (Median 1, Mittelwert über 1)
    z, rho, mig = draws(par, 4000, 7)
    assert z.shape == (4000, 20, 26) and rho.shape == mig.shape == (4000, 20)
    assert rho.mean() == pytest.approx(1.0, abs=0.005) and mig.mean() == pytest.approx(1.0, abs=0.01)
    assert np.median(np.exp(par.sig_meas * z)) == pytest.approx(1.0, abs=0.01) and np.exp(par.sig_meas * z).mean() == pytest.approx(np.exp(par.sig_meas ** 2 / 2), abs=0.01)
    assert "np.exp(par.sig_meas * z" in inspect.getsource(simulate)
    # Erstes Jahr: nur Zuwanderer (Dichte m, Resistenzanteil p); ohne Wachstum bleibt das eine Woche lang so
    res = simulate(Par(W=1, Y=1, r=0.0), "cal", never(), 1, 0, noise=False)
    assert res["resA"][0, 0, 0] == pytest.approx(p, rel=1e-9)
    assert res["net"][0, 0, 0] == pytest.approx(100 * (1 - 0.01 / (0.01 + 0.25)), rel=1e-12)
    # Refugium: der Wochenschaden ist das nach Flächenanteil gewichtete Mittel, die Spritzkosten gelten für die behandelte Fläche
    pq = Par(W=1, Y=1, r=0.0, q=0.5)
    res = simulate(pq, "cal", dict(k=np.array([1]), w0=np.array([0])), 1, 0, noise=False)
    n_beh, n_ref = 0.01 * float(survival(pq)[0] @ immigrant_freq(pq)), 0.01
    expected = 100 * (1 - 0.5 * n_beh / (n_beh + 0.25) - 0.5 * n_ref / (n_ref + 0.25)) - 4.0 * 0.5
    assert res["net"][0, 0, 0] == pytest.approx(expected, rel=1e-9)
    # Gitter und Wertebereiche der Regeln
    cg, pg = R.cal_grid(), R.pause_grid()
    assert len(cg["k"]) == 101 and sorted(set(cg["k"].tolist())) == list(range(1, 14)) + [16, 20, 26, 1000]
    assert all(cg["w0"][cg["k"] == k].tolist() == list(range(min(k, 8))) for k in (1, 7, 8, 26))
    assert len(pg["k"]) == 225 and (set(pg["k"].tolist()), set(pg["w0"].tolist()), set(pg["cap"].tolist()), set(pg["per"].tolist())) == ({1, 2, 4}, {0, 2, 4}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 6})
    assert len(R.s_grid()) == 36 and R.s_grid()[0] == pytest.approx(0.003) and R.s_grid()[-1] == pytest.approx(1.5) and len(R.s_grid(2)) == 18
    src = inspect.getsource(R.tune_rules)
    assert "cap=[1, 2, 3, 4, 6]" in src and "Ysw=[1, 2, 3, 4, 6, 8, 12, 20]" in src and "x <= 2" in src
    # Rückwärtsrechnung: Gitter, Instanz, Rand und Endwert
    assert (DP.GRID1.lo, DP.GRID1.hi, DP.GRID1.step) == (-9.0, 0.7, 0.05) and (DP.GRID2.lo, DP.GRID2.hi, DP.GRID2.step) == (-8.0, 0.6, 0.1)
    d1, d2 = DP.det_par(par), DP.det_par2(par)
    assert (d1.q, d1.c_scout, d1.pB0, d1.rr_null, d1.disc) == (0.0, 0.0, 0.0, True, 1.0) and (d2.pB0, d2.rr_null) == (1e-5, True)
    n = DP.GRID1.axis.size
    V = np.add.outer(np.arange(n, dtype=float), 10.0 * np.arange(n))
    assert DP.interp(V, np.array([1e3]), np.array([1e-12]))[0] == V[-1, 0]  # außerhalb des Gitters: nächster Randpunkt
    assert DP.value_at(np.zeros((n, n)), Par()) == 2000.0  # ausgewiesener Wert = 100 · 20 plus Wert der Rückwärtsrechnung
    import sre_grid as grid_module

    assert "range(1, 27)" in inspect.getsource(grid_module) and "min(k, 12)" in inspect.getsource(grid_module)  # Kalendergitter der Referenz (k = 1 bis 26, Start 0 bis min(k, 12) − 1)
    for text in ("aus den gerundeten Werten ergäben sich 11,2 %", "k = 1 bis 26 und Start 0 bis min(k, 12) − 1", "Gitterfehler 0,5 bei zwei Wirkstoffen gegen 0,1 bei einem", "13 Kontrollwochen",
                 "2000 Pfade in den Regeltabellen, 1000 in Raster und Horizonten", "bei Gleichstand der Summen gilt der erste Eintrag des Gitters", "ohne den Streufaktor μ", "alle drei Schwellen (auf 1, 5 und 20 Jahre abgestimmt)",
                 "kein großer Hebel", "das Größere von 1,5 Punkten und vier Standardfehlern", "Planungshorizont 20 beim Preset „Kurzer Horizont“"):
        assert text in readme, text
    assert round(100 * (1123.5 / 1009.9 - 1), 1) == 11.2 and len(range(0, par.W, 2)) == 13  # gerundet 11,2 %, 13 Kontrollwochen bei Kontrollabstand 2
    assert C.PRESETS["Kurzer Horizont (Zins 0,8)"] == {"disc": 0.8} and C.DEFAULT_HORIZON == 20
    # Gleichstand: der erste Eintrag des Gitters gewinnt (np.argmax), beim Kontrollabstand der kleinste (strenges „>“)
    tie, _ = R.tune(par, "cal", H=1, P=20, grid=dict(k=np.array([1000, 1000]), w0=np.array([5, 7])))
    assert int(tie["w0"][0]) == 5
    _, v_tie, _ = R.tune_v(Par(c_scout=0.0), H=1, grid=dict(S=np.array([1e9])), P=20)
    assert v_tie == 1
    V1, _ = DP.solve(Par(W=1, Y=1), years=1)
    assert V1[0, 0] == pytest.approx(-DP.week_step(Par(W=1, Y=1), 1e-9, 1e-9, 0)[2], rel=1e-9)  # Endwert nach der letzten Woche ist 0


def test_readme_nennt_die_zahlen_der_tabellen(grund):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    zeile = " / ".join(str(round(grund[n]["total"])) for n in (C.R_NEVER, C.R_CAL, C.R_THR1, C.R_THR5, C.R_THRH))
    assert zeile in readme
    for text in ("1123,5", "1009,9", "816,1", "983,3", "1607,1", "1607,6", "250000/2601", "91/1091", "19 von 42", "37 Zellen", "17 Zellen", "in allen 6 vorgerechneten Szenarien",
                 "348 / 684 / 684 / 684 / 699 / 811", "792 / 811 / 821 / 829", "1533 / 1524 / 1472 / 1351 / 1274", "796 bei S = 0,87", "11,3 % darüber",
                 "im Grundfall der stärkste Hebel unter den verglichenen Regeln", "rund 13-mal je Jahr spritzt (11,8 im Mittel aller 20 Jahre; Mittel über die 2000 Auswertungspfade", "727 bei S = 1,5 (0,4 Spritzungen je Jahr",
                 "spritzt rund 1,8-mal je Jahr", "Wochen 0 bis 3 und 5 in den Jahren 5 und 8, Wochen 0 bis 4 danach", "in den Presets bis 2,9", "Abstand 71", "Abstand 6", "weiter 50 Punkte unter",
                 "auf benachbarten Gitterwerten (0,006 und 0,007)", "sinkt der Kalender um 6, die Kurzsicht-Schwelle um 16 Punkte"):
        assert text in readme, text


def test_readme_testzahl_stimmt_mit_der_suite():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True, timeout=300).stdout
    n = int(re.search(r"(\d+) tests? collected", out).group(1))
    assert f"{n} Tests (`pytest --collect-only -q tests`)" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_vorgerechnete_tabelle_stimmt_mit_den_presets_ueberein(pre):
    for name in C.PRESETS:
        par = par_from_settings(settings_of(name))
        assert par.disc == pre[name]["settings"]["disc"] and par.phi == pre[name]["settings"]["phi"]
