"""Presets: gültige Werte, Hilfetexte, Standard-Preset = Standardeinstellungen, Schlüssel gedeckt, jedes Preset einzigartig, Abgleich mit den vorgerechneten Szenarien."""

import pytest

import sre_constants as C
import sre_presets as P
from sre_model import Par
from sre_scenario import DEFAULTS, TABLE_KEYS, par_from_settings, settings_of, table_key


def test_every_preset_has_help():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    assert all(C.PRESET_HELP[n] for n in C.PRESETS)


def test_preset_keys_are_known_settings():
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name


def test_preset_values_are_inside_the_slider_bounds():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            assert spec.lo <= value <= spec.hi, (name, key, value)


def test_default_preset_changes_nothing():
    assert C.PRESETS["Grundfall (φ 0,9, c 0,3)"] == {}
    assert list(C.PRESETS)[0] == "Grundfall (φ 0,9, c 0,3)"


def test_setting_specs_cover_every_preset_key_and_every_default():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert {spec.url_param: spec.default for spec in P.SETTING_SPECS.values()} == DEFAULTS


def test_defaults_inside_bounds():
    for key, spec in P.SETTING_SPECS.items():
        assert spec.lo <= spec.default <= spec.hi, key


def test_no_two_presets_set_the_same_values():
    seen = {}
    for name in C.PRESETS:
        full = tuple(sorted(settings_of(name).items()))
        assert full not in seen, (name, seen.get(full))
        seen[full] = name


def test_grundfall_parameters_are_the_model_defaults():
    assert par_from_settings(DEFAULTS) == Par()


def test_presets_change_exactly_what_their_name_says():
    base = par_from_settings(DEFAULTS)
    billig = par_from_settings(settings_of("Resistenz billig und stark (φ 1,0, c 0,1)"))
    assert (billig.phi, billig.cost_fit) == (1.0, 0.1) and billig.cross == base.cross and billig.disc == base.disc
    assert par_from_settings(settings_of("Resistenz teuer (c 0,5)")).cost_fit == 0.5
    assert par_from_settings(settings_of("Resistenz schwach (φ 0,5)")).phi == 0.5
    assert par_from_settings(settings_of("Kreuzresistenz 0,5")).cross == 0.5
    assert par_from_settings(settings_of("Kurzer Horizont (Zins 0,8)")).disc == 0.8


def test_rr_regler_standard_ist_das_produkt_und_sonst_die_zehnerpotenz():
    assert par_from_settings({**DEFAULTS, "rr": -10}).rr_freq == -1.0
    assert par_from_settings({**DEFAULTS, "rr": -4}).rr_freq == pytest.approx(1e-4)
    assert par_from_settings({**DEFAULTS, "rr": -3}).rr_freq == pytest.approx(1e-3)


def test_table_key_ignores_seed_and_paths_only():
    a = {**DEFAULTS, "seed": 5, "paths": 500}
    assert table_key(a) == table_key(DEFAULTS)
    for k in TABLE_KEYS:
        assert table_key({**DEFAULTS, k: DEFAULTS[k] + 1}) != table_key(DEFAULTS), k


def test_vorgerechnete_szenarien_gehoeren_zu_den_presets(pre):
    assert list(pre) == list(C.PRESETS)
    for name, sc in pre.items():
        assert table_key(sc["settings"]) == table_key(settings_of(name)), name
        assert [r["rule"] for r in sc["rows"]] == list(C.RULES), name


def test_zahlen_der_annahmentexte_sind_die_vorgaben_des_modells():
    """Die Zahlen im Text „Annahmen“ (App und README) sind die Standardwerte des Modells."""
    import sre_texts as TX

    p = Par()
    assert (p.r, p.h, p.kill, p.nu, p.m, p.p0, p.sig_w, p.sig_mig, p.K, p.W, p.Y) == (0.45, 0.25, 0.9, 0.01, 0.01, 1e-5, 0.15, 0.5, 1.0, 26, 20)
    for fragment in ("Wachstumsrate 0,45", "Halbsättigung 0,25", "Abtötungsrate 0,9", "Überwinterung 0,01", "Zuwanderung 0,01", "10⁻⁵", "Wetterstreuung 0,15", "Streuung der Zuwanderung 0,5",
                     "Resistenzgrad 0,9, Fitnesskosten 0,3 und Kontrollkosten 0,1", "Resistenzgrad 0,95, Fitnesskosten 0,1, Kontrollkosten 0,3"):
        assert fragment in TX.ASSUMPTIONS_MARKDOWN, fragment
    assert (p.phi, p.cost_fit, p.c_scout, p.c_spray, p.sig_meas) == (0.9, 0.3, 0.1, 4.0, 0.3)
    assert C.P_TRAIN == 400 and C.P_EVAL == 2000 and C.P_TRAIN_QUICK == 60 and C.SEED_TRAIN == 11 and C.SEED_EVAL == 22
