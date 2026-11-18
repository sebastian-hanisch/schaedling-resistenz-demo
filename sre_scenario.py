"""Einstellungen (Regler und Presets) <-> Modellparameter. Keine Streamlit-Abhängigkeit: auch `sre_grid.py` und die Tests nutzen es."""

import sre_constants as C
from sre_model import Par

DEFAULTS = {
    "phi": C.DEFAULT_PHI, "cost": C.DEFAULT_COST, "kill": C.DEFAULT_KILL, "m": C.DEFAULT_M, "spray": C.DEFAULT_SPRAY, "scout": C.DEFAULT_SCOUT,
    "meas": C.DEFAULT_MEAS, "cross": C.DEFAULT_CROSS, "rr": C.DEFAULT_RR, "q": C.DEFAULT_Q, "horizon": C.DEFAULT_HORIZON, "disc": C.DEFAULT_DISC,
    "seed": C.DEFAULT_SEED, "paths": C.DEFAULT_PATHS,
}
# Einstellungen, von denen die vorgerechneten Tabellen abhängen (nicht Zufalls-Seed und Pfadzahl)
TABLE_KEYS = ("phi", "cost", "kill", "m", "spray", "scout", "meas", "cross", "rr", "q", "horizon", "disc")


def settings_of(preset: str) -> dict:
    """Alle Einstellungen eines Presets (was es nicht nennt, hat den Standardwert)."""
    return {**DEFAULTS, **C.PRESETS[preset]}


def par_from_settings(s: dict) -> Par:
    """Modellparameter zu den Reglerwerten. Der Refugium-Anteil q steckt NICHT in Par: er gilt nur für die zwei Refugium-Zeilen (siehe `sre_rules.tune_rules`).
    rr = Zehnerpotenz des Anteils Doppelresistenter; der Standard −10 heißt: Produkt der Einzelanteile (10^-5 · 10^-5)."""
    rr = int(s["rr"])
    return Par(phi=float(s["phi"]), cost_fit=float(s["cost"]), kill=float(s["kill"]), m=float(s["m"]), c_spray=float(s["spray"]),
               c_scout=float(s["scout"]), sig_meas=float(s["meas"]), cross=float(s["cross"]), rr_freq=-1.0 if rr <= C.DEFAULT_RR else 10.0 ** rr,
               disc=float(s["disc"]))


def table_key(s: dict) -> tuple:
    """Schlüssel zum Vergleich mit den vorgerechneten Szenarien (gerundet, damit Gleitkomma-Reste nicht stören)."""
    return tuple(round(float(s[k]), 6) for k in TABLE_KEYS)
