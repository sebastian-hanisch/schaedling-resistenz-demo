"""Verdikte (reine Funktionen in sre_evaluation.py) an handgesetzten Werten: jede Schwelle und jede Zahl im Text."""

import pytest

import sre_evaluation as ev


def test_pct_und_nenner_null():
    assert ev.pct(5.0, 100.0) == pytest.approx(5.0)
    assert ev.pct(1.0, 0.0) is None


# --------------------------------------------------------------------------------------------------------- Kurzsicht
def test_kurzsicht_unter_nie_ist_eine_warnung_und_nennt_die_zahlen():
    kind, text = ev.kurzsicht_verdikt(349.4, 728.2, 811.0)
    assert kind == "warning" and "Kurzsicht kostet mehr als Nichtstun" in text
    assert "349" in text and "728" in text and "811" in text and "Unterschied 462" in text


def test_kurzsicht_grenze_blind_gleich_nie_ist_keine_warnung():
    assert ev.kurzsicht_verdikt(728.0, 728.0, 740.0)[0] != "warning"  # genau gleich: nicht „mehr als Nichtstun“
    assert ev.kurzsicht_verdikt(727.9, 728.0, 740.0)[0] == "warning"


def test_kurzsicht_schwelle_genau():
    assert ev.kurzsicht_verdikt(700.0, 700.0, 735.0)[0] == "info"      # Abstand genau 35 = 5 % von nie: „spürbar“
    assert ev.kurzsicht_verdikt(700.1, 700.0, 735.0)[0] == "success"   # Abstand 34,9
    kind, text = ev.kurzsicht_verdikt(1253.0, 728.0, 1324.0)
    assert kind == "info" and "Kurzsicht kostet spürbar" in text and "71 Ertragspunkte" in text
    kind, text = ev.kurzsicht_verdikt(1496.0, 728.0, 1489.0)
    assert kind == "success" and "7 Ertragspunkte" in text and "Kein Gedächtnisproblem" in text  # Betrag des Abstands


# --------------------------------------------------------------------------------------------------------- Pausen
def test_pausen_im_rauschen_bis_drei_standardfehler_inklusive():
    kind, text = ev.pausen_verdikt(3.0, 1.0, 1000.0)  # genau 3 SE
    assert kind == "info" and "im Rauschen" in text and "+3" in text and "1.0" in text
    assert "Rauschen" not in ev.pausen_verdikt(3.1, 1.0, 1000.0)[1]  # 3,1 Standardfehler: nicht mehr im Rauschen
    assert "Rauschen" in ev.pausen_verdikt(-3.0, 1.0, 1000.0)[1]  # Betrag zählt


def test_pausen_deutlich_knapp_schaden():
    kind, text = ev.pausen_verdikt(103.0, 0.3, 979.0)
    assert kind == "success" and "lohnen deutlich" in text and "103 Ertragspunkte" in text and "10.5 %" in text
    kind, text = ev.pausen_verdikt(50.0, 0.3, 1000.0)  # genau 5 %
    assert kind == "success"
    kind, text = ev.pausen_verdikt(49.9, 0.3, 1000.0)
    assert kind == "info" and "lohnen knapp" in text and "5.0 %" in text
    kind, text = ev.pausen_verdikt(-20.0, 0.3, 1000.0)
    assert kind == "warning" and "schaden" in text and "20 Ertragspunkte" in text and "2.0 %" in text
    assert "nicht möglich" in ev.pausen_verdikt(5.0, 0.3, 0.0)[1]


# --------------------------------------------------------------------------------------------------------- Wechsel
def test_wechsel_schwellen_und_text():
    kind, text = ev.wechsel_verdikt(1533.0, 1082.0, 0.0)
    assert kind == "success" and "Wirkstoffwechsel hilft deutlich" in text and "stärkste" not in text and "451 Ertragspunkte" in text and "41.7 %" in text and "Kreuzresistenz" not in text
    assert "Kreuzresistenz 0.5" in ev.wechsel_verdikt(1533.0, 1082.0, 0.5)[1]
    assert ev.wechsel_verdikt(1050.0, 1000.0, 0.0)[0] == "success"   # genau 5 %
    kind, text = ev.wechsel_verdikt(1049.9, 1000.0, 0.0)
    assert kind == "info" and "hilft wenig" in text and "5.0 %" in text
    assert ev.wechsel_verdikt(1010.1, 1000.0, 0.0)[0] == "info"       # über 1 %
    kind, text = ev.wechsel_verdikt(1010.0, 1000.0, 0.0)              # genau 1 %: nicht mehr „hilft wenig“
    assert kind == "warning" and "bringt hier nichts" in text
    kind, text = ev.wechsel_verdikt(973.0, 979.0, 1.0)
    assert kind == "warning" and "-6 Ertragspunkte" in text and "Kreuzresistenz 1" in text
    assert "nicht möglich" in ev.wechsel_verdikt(5.0, 0.0, 0.0)[1]


# --------------------------------------------------------------------------------------------------------- Voraussicht
def test_voraussicht_schwellen_und_zahlen():
    kind, text = ev.voraussicht_verdikt(1123.5, 1009.9, 983.3)
    assert kind == "info" and "Voraussicht lohnt:" in text and "11.2 %" in text and "1124" in text and "1010" in text and "983" in text
    assert ev.voraussicht_verdikt(1050.0, 1000.0, 900.0)[0] == "info" and "lohnt:" in ev.voraussicht_verdikt(1050.0, 1000.0, 900.0)[1]  # genau 5 %
    kind, text = ev.voraussicht_verdikt(1049.9, 1000.0, 900.0)
    assert kind == "info" and "lohnt knapp" in text
    assert "knapp" in ev.voraussicht_verdikt(1010.0, 1000.0, 900.0)[1]                                                                   # genau 1 %
    kind, text = ev.voraussicht_verdikt(1009.9, 1000.0, 900.0)
    assert kind == "success" and "kaum etwas" in text
    assert "nicht möglich" in ev.voraussicht_verdikt(5.0, 0.0, 0.0)[1]


# --------------------------------------------------------------------------------------------------------- Kipp
def test_kipp_schwelle_inklusive_und_zahlen():
    kind, text = ev.kipp_verdikt(0.9, 0.3, 0.006, 0.25, 348.0, 797.0)
    assert kind == "warning" and "Gedächtnis-Dilemma" in text and "φ = 0.9" in text and "c = 0.3" in text and "0.006" in text and "0.250" in text and "348" in text and "797" in text
    kind, text = ev.kipp_verdikt(0.9, 0.5, 0.006, 0.249, 1250.0, 1256.0)
    assert kind == "success" and "Kein Gedächtnisproblem im Raster" in text
    assert "Summen liegen 6 Ertragspunkte auseinander" in text and "unter 0.25" in text  # Betrag des Abstands der angezeigten Summen, keine Gleichheit behauptet
    assert "nahe beieinander" not in text


def test_lueckentext_herkunft():
    assert "400 Trainingspfaden" in ev.lueckentext(0, 0, 0, "vorgerechnet") and "2000" in ev.lueckentext(0, 0, 0, "vorgerechnet")
    t = ev.lueckentext(12.4, 60, 200, "schnell")
    assert "12 s" in t and "60 Trainingspfaden" in t and "200 getrennten" in t
