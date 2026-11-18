"""Kern der Schädlings-Demo: Saison-Simulation mit vier Genotypen (alle Zahlen sind Annahmen).

Eine Saison hat `W` Wochen, die Dichte ist relativ zur Kapazität K = 1. Vier Genotypen vermehren sich klonal:
SS (anfällig), RS (resistent gegen Wirkstoff A), SR (gegen B), RR (gegen beide); die Reihenfolge in allen Feldern ist SS, RS, SR, RR.

Reihenfolge innerhalb einer Woche (jede Definition ist mit einem Randfalltest abgesichert, tests/test_edge_cases.py):
1. Die Regel entscheidet anhand der Dichte der behandelten Fläche zu Wochenbeginn (Schwelle: gemessen, nur in Kontrollwochen t mod v = 0).
2. Spritzung tötet VOR dem Wachstum: Anfällige zum Anteil `kill`, Resistente zum Anteil kill·(1 − φ); Kosten c_spray·(1 − q).
3. Wachstum nach Ricker: n_g · exp(r·ρ·f_g·(1 − N/K)), N = Gesamtdichte NACH der Spritzung, f_g = (1 − c)^(Zahl der Resistenzen), ρ = Wetter der Saison.
4. Wochenschaden = 100·b/W · N̄/(N̄ + h) mit N̄ = Mittel der Gesamtdichte vor (nach der Spritzung) und nach dem Wachstum.
Ertrag je Jahr = 100 − Summe der Wochenschäden − Kosten (Ertragspunkte). Am Saisonende (Pool aus behandelter Fläche und Refugium) folgt der Winter:
neue Anfangsdichte = ν · Pool + m · (Zuwanderung, lognormal gestreut) · (Genotypanteile der Zuwanderer). Das erste Jahr startet NUR mit Zuwanderern
(ein neuer Wirkstoff: Resistenz nur als Hintergrund p0); das ist ein Szenario, kein Gleichgewicht.
"""

from dataclasses import dataclass

import numpy as np

GENOTYPES = ("SS", "RS", "SR", "RR")


@dataclass(frozen=True)
class Par:
    """Annahmen des Modells; die Standardwerte sind der Grundfall (nach ersten Läufen gesetzt, siehe README „Ehrliche Grenzen“)."""

    W: int = 26                 # Wochen je Saison
    Y: int = 20                 # Jahre
    r: float = 0.45             # Wachstumsrate je Woche (Ricker)
    K: float = 1.0              # Kapazität
    b: float = 1.0              # größter Ertragsverlust je Saison als Anteil (Schaden gesättigt)
    h: float = 0.25             # Halbsättigung des Schadens (Holling Typ II)
    nu: float = 0.01            # Überwinterung: Anteil des Pools
    m: float = 0.01             # Zuwanderung je Winter (Dichte)
    p0: float = 1e-5            # Resistenzanteil der Zuwanderer je Wirkstoff
    pB0: float = -1.0           # Anteil gegen B (-1 = wie p0)
    rr_null: bool = False       # True: keine Doppelresistenten in der Zuwanderung (nur für die Rückwärtsrechnung)
    rr_freq: float = -1.0       # Anteil Doppelresistenter (-1 = Produkt der Einzelanteile)
    cross: float = 0.0          # Kreuzresistenz: Anteil des Resistenzschutzes, der auch gegen den anderen Wirkstoff gilt
    kill: float = 0.9           # Abtötungsrate (Anfällige)
    phi: float = 0.9            # Resistenzgrad
    cost_fit: float = 0.3       # Fitnesskosten je Resistenz (Wachstum)
    c_spray: float = 4.0        # Kosten einer Spritzung (Ertragspunkte)
    c_scout: float = 0.1        # Kosten einer Kontrolle (Ertragspunkte)
    sig_w: float = 0.15         # Streuung des Wetterfaktors (lognormal)
    sig_mig: float = 0.5        # Streuung der Zuwanderung (lognormal)
    sig_meas: float = 0.3       # Messrauschen (lognormal)
    q: float = 0.0              # Refugium: Flächenanteil, der nie gespritzt wird
    disc: float = 1.0           # Zinsfaktor je Jahr (1 = keine Abzinsung)


def with_(par: Par, **changes) -> Par:
    """Kopie mit geänderten Feldern."""
    return Par(**{**par.__dict__, **changes})


def fitness(par: Par) -> np.ndarray:
    c = par.cost_fit
    return np.array([1.0, 1 - c, 1 - c, (1 - c) ** 2])


def survival(par: Par) -> np.ndarray:
    """Überlebensanteil je Wirkstoff (Zeile 0 = A, 1 = B) und Genotyp (SS, RS, SR, RR)."""
    s_sus = 1 - par.kill
    s_res = 1 - par.kill * (1 - par.phi)
    s_x = s_sus + par.cross * (s_res - s_sus)  # Resistenz gegen den anderen Wirkstoff schützt nur zum Anteil `cross`
    return np.array([[s_sus, s_res, s_x, s_res],
                     [s_sus, s_x, s_res, s_res]])


def immigrant_freq(par: Par) -> np.ndarray:
    """Genotypanteile der Zuwanderer (SS, RS, SR, RR)."""
    p = par.p0
    pb = par.p0 if par.pB0 < 0 else par.pB0
    if par.rr_null:
        return np.array([1 - p - pb, p, pb, 0.0])
    if par.rr_freq >= 0:
        return np.array([1 - p - pb - par.rr_freq, p, pb, par.rr_freq])
    return np.array([(1 - p) * (1 - pb), p * (1 - pb), (1 - p) * pb, p * pb])


def draws(par: Par, paths: int, seed: int, years=None, noise=True):
    """Zufallszahlen eines Laufs: Messrauschen z[P, Y, W], Wetter ρ[P, Y], Zuwanderung mig[P, Y] (Mittelwert 1).
    Gleiche Seeds geben allen Regeln dieselben Pfade (gepaarte Unterschiede)."""
    Y = years or par.Y
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((paths, Y, par.W))
    w = rng.standard_normal((paths, Y))
    g = rng.standard_normal((paths, Y))
    if not noise:
        return np.zeros_like(z), np.ones((paths, Y)), np.ones((paths, Y))
    rho = np.exp(par.sig_w * w - 0.5 * par.sig_w ** 2)
    mig = np.exp(par.sig_mig * g - 0.5 * par.sig_mig ** 2)
    return z, rho, mig


def simulate(par: Par, kind: str, prm: dict, paths: int, seed: int, years=None, mode="A", noise=True, v=1, record=False):
    """Vektorisiert über G Parametersätze (je Eintrag in `prm`) und `paths` Pfade.

    kind "cal": prm["k"], prm["w0"] (Arrays der Länge G): spritze in den Wochen w0, w0 + k, ... (k = 1000: nie).
    kind "thr": prm["S"]: spritze, wenn die gemessene Dichte S überschreitet (Kontrolle alle `v` Wochen, Kosten c_scout je Kontrolle).
    Optional in `prm`: "cap" (höchstens so viele Spritzungen je Saison) und "per" (nur in Jahren mit Index mod per = 0, Jahr 1 hat Index 0).
    mode: "A" nur Wirkstoff A; "year" A und B je Jahr im Wechsel (Jahr 1 mit A); "spray" je Spritzung im Wechsel (Zähler über alle Jahre);
          "seq" A bis zum Jahr prm["Ysw"] (ausschließlich), danach B.
    Rückgabe: dict mit net[G,P,Y] (Ertragspunkte), sprays[G,P,Y], resA, resB (Resistenzanteil im Pool am Saisonende), endN, loss;
    mit record=True zusätzlich trace_n[G,Y,W,4] (mittlere Dichte je Genotyp am Wochenende), trace_spray[G,Y,W], trace_loss[G,Y,W]."""
    Y = years or par.Y
    W = par.W
    G = len(next(iter(prm.values())))
    z, rho, mig = draws(par, paths, seed, Y, noise)
    fit = fitness(par)
    surv = survival(par)
    imm = immigrant_freq(par)
    q = par.q
    area = np.array([1 - q, q])
    P = paths
    n = np.zeros((G, P, 2, 4))  # [Parametersatz, Pfad, Fläche (0 behandelt, 1 Refugium), Genotyp]
    n[:] = (par.m * imm)[None, None, None, :]
    spray_count = np.zeros((G, P))
    net = np.zeros((G, P, Y))
    sprays = np.zeros((G, P, Y))
    resA = np.zeros((G, P, Y))
    resB = np.zeros((G, P, Y))
    endN = np.zeros((G, P, Y))
    lossY = np.zeros((G, P, Y))
    if record:
        trace_n = np.zeros((G, Y, W, 4))
        trace_spray = np.zeros((G, Y, W))
        trace_loss = np.zeros((G, Y, W))
    for y in range(Y):
        loss = np.zeros((G, P))
        cost = np.zeros((G, P))
        nsp = np.zeros((G, P))
        for t in range(W):
            Nt = n[:, :, 0, :].sum(-1)
            if kind == "thr":
                if t % v == 0:
                    nhat = Nt * np.exp(par.sig_meas * z[None, :, y, t]) if noise else Nt
                    cost += par.c_scout
                    spray = nhat > prm["S"][:, None]
                else:
                    spray = np.zeros((G, P), dtype=bool)
            else:
                k = prm["k"][:, None]
                w0 = prm["w0"][:, None]
                spray = (t >= w0) & (((t - w0) % k) == 0) & np.ones((G, P), dtype=bool)
            if "cap" in prm:
                spray = spray & (nsp < prm["cap"][:, None])
            if "per" in prm:
                spray = spray & ((y % prm["per"][:, None]) == 0)
            if mode == "A":
                ing = np.zeros((G, P), dtype=int)
            elif mode == "year":
                ing = np.full((G, P), y % 2, dtype=int)
            elif mode == "spray":
                ing = (spray_count % 2).astype(int)
            elif mode == "seq":
                ing = (y >= prm["Ysw"][:, None]).astype(int) * np.ones((G, P), dtype=int)
            else:
                raise ValueError(mode)
            sv = surv[ing]  # (G, P, 4)
            n[:, :, 0, :] = np.where(spray[..., None], n[:, :, 0, :] * sv, n[:, :, 0, :])
            cost += spray * par.c_spray * (1 - q)
            nsp += spray
            spray_count += spray
            N1 = n.sum(-1)  # (G, P, 2)
            growth = np.exp(par.r * rho[None, :, y, None, None] * fit[None, None, None, :] * (1 - N1[..., None] / par.K))
            n = n * growth
            N2 = n.sum(-1)
            Nm = 0.5 * (N1 + N2)
            wk = par.b / W * Nm / (Nm + par.h)
            week_loss = (wk * area[None, None, :]).sum(-1)
            loss += week_loss
            if record:
                trace_n[:, y, t, :] = n[:, :, 0, :].mean(1)
                trace_spray[:, y, t] = spray.mean(1)
                trace_loss[:, y, t] = 100 * week_loss.mean(1)
        pool = (1 - q) * n[:, :, 0, :] + q * n[:, :, 1, :]
        Npool = pool.sum(-1)
        resA[:, :, y] = (pool[..., 1] + pool[..., 3]) / Npool
        resB[:, :, y] = (pool[..., 2] + pool[..., 3]) / Npool
        endN[:, :, y] = Npool
        net[:, :, y] = 100 * (1 - loss) - cost
        sprays[:, :, y] = nsp
        lossY[:, :, y] = loss
        n0 = par.nu * pool + par.m * mig[None, :, y, None] * imm[None, None, :]
        n = np.repeat(n0[:, :, None, :], 2, axis=2).copy()
    out = dict(net=net, sprays=sprays, resA=resA, resB=resB, endN=endN, loss=lossY)
    if record:
        out.update(trace_n=trace_n, trace_spray=trace_spray, trace_loss=trace_loss)
    return out


def total(res: dict, par: Par, upto=None) -> np.ndarray:
    """Summe der (mit dem Zinsfaktor abgezinsten) Jahresergebnisse über die ersten `upto` Jahre; Form (G, P)."""
    net = res["net"]
    Y = net.shape[2] if upto is None else upto
    w = par.disc ** np.arange(Y)
    return (net[:, :, :Y] * w[None, None, :]).sum(-1)


def mean_se(x: np.ndarray):
    """Mittel und Standardfehler über die letzte Achse (Pfade)."""
    return x.mean(-1), x.std(-1, ddof=1) / np.sqrt(x.shape[-1])


def never() -> dict:
    """Regel „nie spritzen“ als Kalender, der nie auslöst."""
    return dict(k=np.array([1000]), w0=np.array([1000]))
