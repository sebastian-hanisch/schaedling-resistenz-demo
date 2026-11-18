"""Konstanten: Regler-Grenzen, Standardwerte (Grundfall), Abstimmungsaufwand, Regelnamen, Farben und Presets.

ALLE Modellzahlen sind Annahmen (nicht aus Messungen an Schädlingen oder Wirkstoffen, keine Literaturwerte). Die Standardwerte wurden nach ersten Läufen gesetzt:
die erste Fassung (φ = 0,95, c = 0,1, Kontrollkosten 0,3) ließ in der Vormessung (anderer Code-Stand) den Wirkstoff schnell versagen und die Schwellenregeln im Kontrollaufwand
untergehen; mit dem heutigen Modell gilt das bei diesen Werten nur für die kurzsichtige Schwelle. Das ist ein Freiheitsgrad der Demo und steht in der README unter „Ehrliche Grenzen“."""

# Regler: Minimum, Maximum, Standard (Standard = Grundfall)
PHI_MIN, PHI_MAX, DEFAULT_PHI = 0.0, 1.0, 0.9            # Resistenzgrad: Resistente werden zum Anteil kill·(1 − φ) getötet
COST_MIN, COST_MAX, DEFAULT_COST = 0.0, 0.6, 0.3          # Fitnesskosten je Resistenz (Wachstum)
KILL_MIN, KILL_MAX, DEFAULT_KILL = 0.5, 0.99, 0.9         # Abtötungsrate der Anfälligen
M_MIN, M_MAX, DEFAULT_M = 0.001, 0.1, 0.01                # Zuwanderung je Winter (Dichte, Kapazität = 1)
SPRAY_MIN, SPRAY_MAX, DEFAULT_SPRAY = 1.0, 10.0, 4.0      # Kosten einer Spritzung (Ertragspunkte)
SCOUT_MIN, SCOUT_MAX, DEFAULT_SCOUT = 0.0, 1.0, 0.1       # Kosten einer Kontrolle (Ertragspunkte)
MEAS_MIN, MEAS_MAX, DEFAULT_MEAS = 0.0, 1.0, 0.3          # Messrauschen (lognormale Streuung)
CROSS_MIN, CROSS_MAX, DEFAULT_CROSS = 0.0, 1.0, 0.0       # Kreuzresistenz
RR_MIN, RR_MAX, DEFAULT_RR = -10, -3, -10                 # Zehnerpotenz des Anteils Doppelresistenter an den Zuwanderern; −10 = Produkt der Einzelanteile
Q_MIN, Q_MAX, DEFAULT_Q = 0.0, 0.5, 0.2                   # Refugium-Anteil (gilt nur für die zwei Refugium-Zeilen)
HORIZON_MIN, HORIZON_MAX, DEFAULT_HORIZON = 1, 20, 20     # Planungshorizont der Abstimmung (Jahre)
DISC_MIN, DISC_MAX, DEFAULT_DISC = 0.5, 1.0, 1.0          # Zinsfaktor je Jahr
SEED_MIN, SEED_MAX, DEFAULT_SEED = 0, 9999, 22            # Zufalls-Seed der Auswertungspfade (22 = der Seed der vorgerechneten Tabellen)
PATHS_MIN, PATHS_MAX, DEFAULT_PATHS = 50, 1000, 200       # Pfadzahl der Auswertung in der App

YEARS = 20                # Jahre jeder Auswertung
WEEKS = 26                # Wochen je Saison

# Abstimmung und Auswertung (getrennte Pfade, gleiche Zufallszahlen für alle Regeln)
SEED_TRAIN = 11
SEED_EVAL = 22
P_TRAIN = 400             # vorgerechnete Tabellen: Trainingspfade der Abstimmung
P_EVAL = 2000             # vorgerechnete Tabellen: Auswertungspfade
P_TRAIN_SWEEP = 250       # vorgerechnete Raster und Horizonte
P_EVAL_SWEEP = 1000
P_TRAIN_QUICK = 60        # Schnellabstimmung in der App (Näherung)
P_EVAL_QUICK = 300
S_GRID_N = 36             # Schwellen: 36 Werte, logarithmisch von 0,003 bis 1,5
S_GRID_LO, S_GRID_HI = 0.003, 1.5
V_OPTIONS = (1, 2, 4)     # Kontrollabstand (Wochen), abgestimmt
V_SWEEP = (2, 4)

# Regeln (Namen wie in der Tabelle und in data/regeln.csv)
R_NEVER = "Nie spritzen"
R_CAL = "Kalender"
R_THR1 = "Schwelle, nur auf 1 Jahr abgestimmt (Kurzsicht)"
R_THR5 = "Schwelle, auf 5 Jahre abgestimmt"
R_THRH = "Schwelle, auf den Planungshorizont abgestimmt"
R_CAP = "Schwelle mit Obergrenze je Saison"
R_PAUSE = "Pausenkalender"
R_CAL_YEAR = "Kalender, Wechsel A/B je Jahr"
R_CAL_SPRAY = "Kalender, Wechsel A/B je Spritzung"
R_THR_YEAR = "Schwelle, Wechsel A/B je Jahr"
R_THR_SPRAY = "Schwelle, Wechsel A/B je Spritzung"
R_SEQ = "Schwelle, A dann B (Wechseljahr abgestimmt)"
R_REF = "Schwelle + Refugium"
R_REF_YEAR = "Schwelle + Refugium + Wechsel A/B je Jahr"
RULES = (R_NEVER, R_CAL, R_THR1, R_THR5, R_THRH, R_CAP, R_PAUSE, R_CAL_YEAR, R_CAL_SPRAY, R_THR_YEAR, R_THR_SPRAY, R_SEQ, R_REF, R_REF_YEAR)
DEFAULT_PLOT_RULES = (R_NEVER, R_CAL, R_THR1, R_THRH, R_PAUSE, R_CAL_YEAR)

# Raster Resistenzgrad x Fitnesskosten (vorgerechnet, Kontrollabstand 2 Wochen)
GRID_PHI = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
GRID_COST = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
HORIZONS = (1, 2, 3, 5, 10, 20)   # Horizonte der Abstimmung (vorgerechnet)
KIPP_S = 0.25                      # „S(20 Jahre) deutlich über S(1 Jahr)“: die auf 20 Jahre abgestimmte Schwelle liegt mindestens bei 25 % der Kapazität

COLORS = ("#4c78a8", "#e45756", "#54a24b", "#f58518", "#b279a2", "#72b7b2", "#9d755d", "#bab0ac")

# Presets: jedes setzt ALLE Regler (was es nicht nennt, geht auf den Standardwert)
PRESETS = {
    "Grundfall (φ 0,9, c 0,3)": {},
    "Resistenz billig und stark (φ 1,0, c 0,1)": {"phi": 1.0, "cost": 0.1},
    "Resistenz teuer (c 0,5)": {"cost": 0.5},
    "Resistenz schwach (φ 0,5)": {"phi": 0.5},
    "Kreuzresistenz 0,5": {"cross": 0.5},
    "Kurzer Horizont (Zins 0,8)": {"disc": 0.8},
}
PRESET_HELP = {
    "Grundfall (φ 0,9, c 0,3)": "Resistenzgrad 0,9, Fitnesskosten 0,3, Zuwanderung 0,01, Spritzkosten 4, Kontrollkosten 0,1, Messrauschen 0,3, keine Kreuzresistenz, keine Abzinsung.",
    "Resistenz billig und stark (φ 1,0, c 0,1)": "Resistente überleben jede Spritzung ganz (φ 1,0) und ihr Wachstum kostet kaum etwas (c 0,1): die Resistenz setzt sich am leichtesten durch.",
    "Resistenz teuer (c 0,5)": "Die Resistenz kostet die Hälfte des Wachstums: die auf 1 und auf 20 Jahre abgestimmte Schwelle liegen im Raster (Kontrollabstand 2) 6 Punkte auseinander, in der Regeltabelle mit frei abgestimmtem Kontrollabstand 71.",
    "Resistenz schwach (φ 0,5)": "Resistente werden noch zur Hälfte der Abtötungsrate getötet (φ 0,5): der Wirkstoff bleibt wirksam, der Vorteil der Pausen und des Wirkstoffwechsels schrumpft.",
    "Kreuzresistenz 0,5": "Die Hälfte des Resistenzschutzes gilt auch gegen den anderen Wirkstoff: der Wirkstoffwechsel verliert einen Teil seines Vorteils.",
    "Kurzer Horizont (Zins 0,8)": "Jedes spätere Jahr zählt mit dem Faktor 0,8 weniger: die Zukunft wird kleingerechnet: die Kurzsicht-Schwelle liegt nicht mehr unter nie spritzen, kostet aber weiter 50 abgezinste Punkte gegen die Abstimmung auf den Horizont.",
}

# Rechenpfad der Tests und der App kürzen (die Modul-Fixture `fast_limits` der Tests setzt kleinere Werte)
LIVE_PROFILE_PATHS = 300   # Pfade des Schwellenprofils in der App
