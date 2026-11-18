"""Handbeispiel in exakten Brüchen (`fractions`) für die Tabelle „Von Hand“.

Drei Teile: (1) die Überlebenstabelle der vier Genotypen, (2) die Auslese durch Spritzen (Anteil Resistenter nach k Spritzungen, geschlossene Formel),
(3) eine Saison mit vier Wochen OHNE Wachstum (r = 0), damit alles rational bleibt: Spritzung, Wochenschaden und Ertrag. Der Wachstumsschritt nach Ricker enthält
exp(...) und ist nicht rational; er steht als eine Woche mit Zahlen (`ricker_week`, sechs Stellen). Das Handbeispiel hat KEINEN Zufall und KEINE Fitnesskosten außer
im Wachstumsschritt; es ist ein Rechenbeispiel, keine Voreinstellung der App."""

import math
from fractions import Fraction as F

KILL = F(9, 10)
PHI = F(9, 10)
H_HALF = F(1, 4)
WEEKS = 4
C_SPRAY = F(4)
S_START = F(1, 100)      # Dichte Anfälliger zu Beginn
R_START = F(1, 10000)    # Dichte der gegen A Resistenten (RS) zu Beginn
GENOS = ("SS", "RS", "SR", "RR")


def survival_exact(kill=KILL, phi=PHI, cross=F(0)):
    """Überlebensanteile {Wirkstoff: {Genotyp: Bruch}}. Wirkstoff A: RS und RR resistent; B: SR und RR; ein nur gegen den anderen Wirkstoff Resistenter
    überlebt mit s_sus + cross · (s_res − s_sus)."""
    s_sus = 1 - kill
    s_res = 1 - kill * (1 - phi)
    s_x = s_sus + cross * (s_res - s_sus)
    return {"A": {"SS": s_sus, "RS": s_res, "SR": s_x, "RR": s_res}, "B": {"SS": s_sus, "RS": s_x, "SR": s_res, "RR": s_res}}


def share_after(p, k, s_sus, s_res):
    """Anteil Resistenter nach k Spritzungen ohne Wachstum: p·sR^k / (p·sR^k + (1 − p)·sS^k)."""
    a = p * s_res ** k
    return a / (a + (1 - p) * s_sus ** k)


def share_start(S=S_START, R=R_START):
    return R / (S + R)


def exact_season(spray_weeks, S=S_START, R=R_START, kill=KILL, phi=PHI, h=H_HALF, weeks=WEEKS, c_spray=C_SPRAY):
    """Eine Saison ohne Wachstum. spray_weeks: Wochen (1-basiert), in denen gespritzt wird. Rückgabe: (Zeilen, Ertrag als Bruch).
    Je Woche: erst Spritzung (S mal 1 − kill, R mal 1 − kill·(1 − φ)), dann N = S + R, Wochenschaden = 100/weeks · N / (N + h) (ohne Wachstum ist N̄ = N)."""
    rows = []
    loss_sum = F(0)
    for week in range(1, weeks + 1):
        sprayed = week in spray_weeks
        if sprayed:
            S = S * (1 - kill)
            R = R * (1 - kill * (1 - phi))
        N = S + R
        loss = F(100, weeks) * N / (N + h)
        loss_sum += loss
        rows.append(dict(week=week, sprayed=sprayed, S=S, R=R, N=N, share=R / N, loss=loss))
    net = 100 - loss_sum - c_spray * len([w for w in spray_weeks if 1 <= w <= weeks])
    return rows, net


def ricker_week(S, R, sprayed, r=0.45, fit_res=0.7, kill=0.9, phi=0.9, h=0.25, weeks=26, b=1.0, K=1.0):
    """Eine Woche mit Wachstum nach Ricker (Gleitkomma): erst Spritzung, dann n·exp(r·f·(1 − N/K)), Schaden aus dem Mittel der Dichten vor und nach dem Wachstum."""
    if sprayed:
        S = S * (1 - kill)
        R = R * (1 - kill * (1 - phi))
    N1 = S + R
    g = 1 - N1 / K
    fS = math.exp(r * g)
    fR = math.exp(r * fit_res * g)
    S2, R2 = S * fS, R * fR
    N2 = S2 + R2
    Nm = 0.5 * (N1 + N2)
    loss = 100.0 * b / weeks * Nm / (Nm + h)
    return dict(S1=S, R1=R, N1=N1, g=g, fS=fS, fR=fR, S2=S2, R2=R2, N2=N2, Nm=Nm, loss=loss)
