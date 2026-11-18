"""Inhalt der App: Tabs, Kernfragen, Expander (per Titel, nie per Index), Handbeispiel, Schnellabstimmung, PDF-Knopf, Formelblöcke, Gleichheit von App- und README-Text."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import sre_constants as C
import sre_presets as P
import sre_texts as TX

pytestmark = pytest.mark.usefixtures("fast_limits")
ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")


def _run(**state):
    a = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        a.session_state[k] = v
    a.run()
    assert not a.exception, [str(e) for e in a.exception]
    return a


@pytest.fixture(scope="module")
def at():
    return _run()


def _texts(a):
    return [e.value for e in list(a.success) + list(a.info) + list(a.warning)]


def test_fuenf_tabs_in_fester_reihenfolge(at):
    assert [t.label for t in at.tabs] == ["✋ Von Hand", "🌱 Eine Saison", "📆 Viele Jahre", "🔭 Kurzsicht und Voraussicht", "⚠️ Grenzen"]


def test_expander_stehen_unter_ihren_titeln(at):
    labels = {e.label for e in at.expander}
    assert {"💬 In einfachen Worten", "🔧 Wie wir das erreichen – vollständiger Methodenvergleich", "📐 Mathematische Formulierung", "⚠️ Annahmen, Grenzen und bekannte Probleme"} <= labels


def test_vier_kernfragen_mit_verdikt_im_grundfall(at):
    texts = _texts(at)
    assert any("Kurzsicht kostet mehr als Nichtstun" in t for t in texts)
    assert any("Pausen lohnen deutlich" in t for t in texts)
    assert any("Wirkstoffwechsel hilft deutlich" in t for t in texts)
    assert any("Voraussicht lohnt" in t for t in texts)
    assert any("Gedächtnis-Dilemma" in t for t in texts)  # Raster: nächste Zelle φ 0,9, c 0,3


def test_grundfall_zahlen_stehen_in_der_tabelle_und_den_verdikten(at):
    texts = " ".join(_texts(at))
    assert "349" in texts and "728" in texts and "811" in texts  # Kurzsicht, nie, Horizont
    assert "103 Ertragspunkte" in texts and "1124" in texts and "1010" in texts


def test_ohne_vorgerechnetes_szenario_fehlen_die_abgestimmten_verdikte_und_ein_hinweis_steht_da():
    a = _run(phi_slider=0.85)
    texts = _texts(a)
    assert any("nicht abgestimmt" in t for t in texts)
    assert not any("Kurzsicht kostet" in t or "Pausen lohnen" in t for t in texts)
    assert any("nicht vorgerechnet" in t for t in texts)  # keine Rückwärtsrechnung für diese φ
    assert not any("Voraussicht lohnt" in t for t in texts)


def test_schnellabstimmung_liefert_tabelle_und_verdikte_fuer_neue_einstellung():
    a = _run(phi_slider=0.85)
    next(b for b in a.button if b.key == "quick_button").click().run(timeout=300)
    assert not a.exception, [str(e) for e in a.exception]
    texts = _texts(a)
    assert any("Pausen" in t for t in texts) and any("Wirkstoffwechsel" in t for t in texts)
    assert any("Schnellabstimmung" in m.value for m in a.markdown)


def test_handbeispiel_zeigt_den_exakten_bruch_und_reagiert():
    from fractions import Fraction as F

    a = _run()
    ertrag = next(m for m in a.metric if m.label == "Ertrag der Saison")
    # Voreinstellung: Spritzung in Woche 2; von Hand: Woche 1 mit N = 101/10000, Wochen 2 bis 4 mit N = 1091/1000000, Schaden 25·N/(N + 1/4)
    net = 100 - F(2525, 2601) - 3 * F(27275, 251091) - 4
    assert ertrag.delta == f"= {net}"
    a.multiselect(key="hand_weeks").set_value([]).run()
    assert next(m for m in a.metric if m.label == "Ertrag der Saison").delta == "= 250000/2601"  # ohne Spritzung


def test_pdf_knopf_ist_da(at):
    assert any("download" in str(type(el)).lower() for el in at.main)


def test_jede_abgesetzte_formel_steht_in_einem_eigenen_markdown_aufruf():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    blocks = re.findall(r'st\.markdown\(r"""\$\$.*?\$\$"""\)', src, flags=re.S)
    assert len(blocks) == 11
    assert all(b.count("$$") == 2 for b in blocks)
    assert src.count("$$") == 2 * len(blocks)  # kein abgesetzter Block außerhalb eines eigenen Aufrufs


def test_formeln_kennen_kein_ungeschuetztes_prozentzeichen():
    src = (ROOT / "app.py").read_text(encoding="utf-8") + TX.MATH_NOTE
    for block in re.findall(r"\$\$.*?\$\$", src, flags=re.S):
        assert "%" not in block.replace("\\%", "")


def test_annahmentext_der_app_steht_in_der_readme():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert TX.ASSUMPTIONS_MARKDOWN.strip() in readme


def test_preset_namen_und_regler_der_app_stimmen_mit_den_konstanten(at):
    keys = {w.key for art in ("slider", "number_input") for w in getattr(at.sidebar, art) if w.key}
    assert keys == set(P.SETTING_SPECS)
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)
