"""Permalink (Begrenzen), Presets und Schnellstart (Standardmuster aus dem Demo-Portfolio)."""

from dataclasses import dataclass

import streamlit as st

import sre_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None


SETTING_SPECS = {
    "phi_slider": SettingSpec("phi", float, C.DEFAULT_PHI, C.PHI_MIN, C.PHI_MAX),
    "cost_slider": SettingSpec("cost", float, C.DEFAULT_COST, C.COST_MIN, C.COST_MAX),
    "kill_slider": SettingSpec("kill", float, C.DEFAULT_KILL, C.KILL_MIN, C.KILL_MAX),
    "m_slider": SettingSpec("m", float, C.DEFAULT_M, C.M_MIN, C.M_MAX),
    "spray_slider": SettingSpec("spray", float, C.DEFAULT_SPRAY, C.SPRAY_MIN, C.SPRAY_MAX),
    "scout_slider": SettingSpec("scout", float, C.DEFAULT_SCOUT, C.SCOUT_MIN, C.SCOUT_MAX),
    "meas_slider": SettingSpec("meas", float, C.DEFAULT_MEAS, C.MEAS_MIN, C.MEAS_MAX),
    "cross_slider": SettingSpec("cross", float, C.DEFAULT_CROSS, C.CROSS_MIN, C.CROSS_MAX),
    "rr_slider": SettingSpec("rr", int, C.DEFAULT_RR, C.RR_MIN, C.RR_MAX),
    "q_slider": SettingSpec("q", float, C.DEFAULT_Q, C.Q_MIN, C.Q_MAX),
    "horizon_slider": SettingSpec("horizon", int, C.DEFAULT_HORIZON, C.HORIZON_MIN, C.HORIZON_MAX),
    "disc_slider": SettingSpec("disc", float, C.DEFAULT_DISC, C.DISC_MIN, C.DISC_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, C.SEED_MIN, C.SEED_MAX),
    "paths_slider": SettingSpec("paths", int, C.DEFAULT_PATHS, C.PATHS_MIN, C.PATHS_MAX),
}
PRESET_KEYS = {spec.url_param: state_key for state_key, spec in SETTING_SPECS.items()}
STATE_OF = {spec.url_param: state_key for state_key, spec in SETTING_SPECS.items()}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.lo is not None:
                value = max(spec.lo, value)
            if spec.hi is not None:
                value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Ein Preset setzt ALLE Regler: was es nicht nennt, geht auf den Standardwert (so wie aktives_preset vergleicht)."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(spec.url_param, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def settings_from_state() -> dict:
    """Die aktuellen Einstellungen als dict mit den URL-Namen (wie `sre_scenario.DEFAULTS`)."""
    return {spec.url_param: st.session_state.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()}


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )
