"""Referenz „volle Voraussicht“: Rückwärtsrechnung (dynamische Optimierung) auf der deterministischen Instanz.

Annahmen der Referenz: kein Rauschen (Wetter 1, Zuwanderung 1), Kontrollen gratis und exakt, ein Refugium gibt es nicht, der Zustand ist bekannt, es gibt keinen Zinsfaktor.
Die Rückwärtsrechnung teilt die Wochendynamik mit der Simulation (`week_step` ist dieselbe Gleichung wie in `sre_model.simulate`): sie belegt die Rechenfehler der
Simulation deshalb nicht; dafür steht der unabhängige Rechenweg im Orakeltest (Aufzählung aller Spritzpläne einer Mini-Instanz, zweite Fassung in reinem Python).

Ein Wirkstoff: Zustand (log10 Anfällige, log10 Resistente), Entscheidung je Woche spritzen ja/nein, Wertfunktion auf einem Gitter, bilineare Interpolation.
Zwei Wirkstoffe: Zustand (log10 S, log10 RA, log10 RB), Entscheidung nichts / A / B, Doppelresistente gibt es nicht (RR = 0).
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from sre_model import Par, survival, with_


@dataclass(frozen=True)
class Grid:
    lo: float = -9.0
    hi: float = 0.7
    step: float = 0.05

    @property
    def axis(self) -> np.ndarray:
        return np.arange(self.lo, self.hi + 1e-9, self.step)


GRID1 = Grid(-9.0, 0.7, 0.05)    # ein Wirkstoff
GRID2 = Grid(-8.0, 0.6, 0.1)     # zwei Wirkstoffe (drei Achsen, deshalb gröber)


def det_par(par: Par) -> Par:
    """Die Instanz der Referenz: kein Refugium, keine Kontrollkosten, nur Wirkstoff A hat Resistenz (RB und RR gleich 0), keine Abzinsung."""
    return with_(par, q=0.0, c_scout=0.0, pB0=0.0, rr_null=True, disc=1.0)


def det_par2(par: Par) -> Par:
    """Die Instanz der Referenz mit zwei Wirkstoffen: wie `det_par`, aber auch gegen B gibt es Hintergrundresistenz (p0); Doppelresistente gibt es nicht (RR = 0)."""
    return with_(par, q=0.0, c_scout=0.0, pB0=(par.p0 if par.pB0 < 0 else par.pB0), rr_null=True, disc=1.0)


def week_step(par: Par, S, R, a):
    """Eine Woche (ein Wirkstoff): (S, R) -> (S2, R2) und Verlust plus Kosten in Ertragspunkten. a = 1: gespritzt (vor dem Wachstum)."""
    if a:
        S = S * (1 - par.kill)
        R = R * (1 - par.kill * (1 - par.phi))
    N1 = S + R
    g = 1 - N1 / par.K
    S2 = S * np.exp(par.r * g)
    R2 = R * np.exp(par.r * (1 - par.cost_fit) * g)
    N2 = S2 + R2
    Nm = 0.5 * (N1 + N2)
    loss = 100.0 * par.b / par.W * Nm / (Nm + par.h)
    return S2, R2, loss + par.c_spray * a


def winter(par: Par, S, R):
    """Winter: Pool mal ν plus Zuwanderung m (ohne Streuung, Anteil Resistenter p0)."""
    return par.nu * S + par.m * (1 - par.p0), par.nu * R + par.m * par.p0


def start_state(par: Par):
    return par.m * (1 - par.p0), par.m * par.p0


def interp(V, S, R, grid: Grid = GRID1):
    x = (np.log10(np.maximum(S, 1e-30)) - grid.lo) / grid.step
    y = (np.log10(np.maximum(R, 1e-30)) - grid.lo) / grid.step
    return map_coordinates(V, [x, y], order=1, mode="nearest")


def solve(par: Par, myopic=False, years=None, grid: Grid = GRID1):
    """Rückwärtsrechnung über W·Y Wochen. myopic=True: die Wertfunktion beginnt jedes Jahr bei 0, nur das laufende Jahr zählt
    (jahresweise bestmögliche Antwort bei bekanntem Zustand: Kurzsicht ohne Messfehler).
    Rückgabe: V (Wert zu Beginn des ersten Jahres auf dem Gitter, Ertragspunkte ohne den Grundwert 100 je Jahr) und diffs[t] = Q(spritzen) − Q(nicht)."""
    Y = years or par.Y
    W = par.W
    T = W * Y
    ax = grid.axis
    nx = len(ax)
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    S = 10.0 ** xx
    R = 10.0 ** yy
    V = np.zeros((nx, nx))
    diffs = np.zeros((T, nx, nx), dtype=np.float32)
    for t in range(T - 1, -1, -1):
        last = t % W == W - 1
        Q = []
        for a in (0, 1):
            S2, R2, c = week_step(par, S, R, a)
            if last:
                cont = 0.0 if myopic else interp(V, *winter(par, S2, R2), grid)
            else:
                cont = interp(V, S2, R2, grid)
            Q.append(-c + cont)
        diffs[t] = (Q[1] - Q[0]).astype(np.float32)
        V = np.maximum(Q[0], Q[1])
    return V, diffs


def value_at(V, par: Par, grid: Grid = GRID1) -> float:
    """Summe über alle Jahre, wenn man im Anfangszustand beginnt: 100·Y plus Wert der Rückwärtsrechnung."""
    S, R = start_state(par)
    return 100.0 * par.Y + float(interp(V, np.array([S]), np.array([R]), grid)[0])


def _interp_scalar(D, S, R, grid: Grid):
    x = (np.log10(max(S, 1e-30)) - grid.lo) / grid.step
    y = (np.log10(max(R, 1e-30)) - grid.lo) / grid.step
    return float(map_coordinates(D.astype(np.float64), [[x], [y]], order=1, mode="nearest")[0])


def forward(par: Par, diffs, years=None, grid: Grid = GRID1):
    """Vorwärtslauf mit der Politik „spritzen, wenn diff > 0“ (diff interpoliert) auf der exakten Dynamik.
    Rückgabe: Gesamtwert (100 je Jahr minus Verluste minus Kosten), Spritzungen je Jahr, Resistenzanteil am Saisonende je Jahr, Spritzwochen je Jahr."""
    Y = years or par.Y
    W = par.W
    S, R = start_state(par)
    total = 0.0
    sprays, res_end, weeks = [], [], []
    for y in range(Y):
        ns, wk = 0, []
        net = 100.0
        for t in range(W):
            a = 1 if _interp_scalar(diffs[y * W + t], S, R, grid) > 0 else 0
            S, R, c = week_step(par, S, R, a)
            net -= c
            ns += a
            if a:
                wk.append(t)
        total += net
        sprays.append(ns)
        res_end.append(R / (S + R))
        weeks.append(wk)
        S, R = winter(par, S, R)
    return total, sprays, res_end, weeks


def threshold_value(par: Par, thr: float, years=None) -> float:
    """Summe einer Schwellenregel auf der deterministischen Instanz (exakte Messung, jede Woche kontrolliert, Kontrollen gratis)."""
    Y = years or par.Y
    S, R = start_state(par)
    total = 0.0
    for _ in range(Y):
        net = 100.0
        for _ in range(par.W):
            a = 1 if (S + R) > thr else 0
            S, R, c = week_step(par, S, R, a)
            net -= c
        total += net
        S, R = winter(par, S, R)
    return total


def calendar_value(par: Par, k: int, w0: int, years=None) -> float:
    """Summe eines Kalenders (Wochen w0, w0 + k, ...) auf der deterministischen Instanz; k = 1000 heißt nie."""
    Y = years or par.Y
    S, R = start_state(par)
    total = 0.0
    for _ in range(Y):
        net = 100.0
        for t in range(par.W):
            a = 1 if (t >= w0 and (t - w0) % k == 0) else 0
            S, R, c = week_step(par, S, R, a)
            net -= c
        total += net
        S, R = winter(par, S, R)
    return total


# ----------------------------------------------------------------------------------------------------- zwei Wirkstoffe
def step2(par: Par, S, RA, RB, a):
    """Eine Woche mit zwei Wirkstoffen (a: 0 nichts, 1 A, 2 B; RR = 0)."""
    sv = survival(par)
    if a > 0:
        i = a - 1
        S = S * sv[i][0]
        RA = RA * sv[i][1]
        RB = RB * sv[i][2]
    N1 = S + RA + RB
    g = 1 - N1 / par.K
    c = par.cost_fit
    S2 = S * np.exp(par.r * g)
    RA2 = RA * np.exp(par.r * (1 - c) * g)
    RB2 = RB * np.exp(par.r * (1 - c) * g)
    N2 = S2 + RA2 + RB2
    Nm = 0.5 * (N1 + N2)
    loss = 100.0 * par.b / par.W * Nm / (Nm + par.h)
    return S2, RA2, RB2, loss + (par.c_spray if a > 0 else 0.0)


def winter2(par: Par, S, RA, RB):
    pa, pb = par.p0, (par.p0 if par.pB0 < 0 else par.pB0)
    return par.nu * S + par.m * (1 - pa - pb), par.nu * RA + par.m * pa, par.nu * RB + par.m * pb


def start_state2(par: Par):
    pa, pb = par.p0, (par.p0 if par.pB0 < 0 else par.pB0)
    return par.m * (1 - pa - pb), par.m * pa, par.m * pb


def interp3(V, S, RA, RB, grid: Grid):
    f = lambda zz: (np.log10(np.maximum(zz, 1e-30)) - grid.lo) / grid.step
    return map_coordinates(V, [f(S), f(RA), f(RB)], order=1, mode="nearest")


def solve2(par: Par, years=None, grid: Grid = GRID2):
    """Rückwärtsrechnung mit zwei Wirkstoffen. Rückgabe: V (Wertfunktion zu Beginn) und die Politik pol[t, i, j, k] (0 nichts, 1 A, 2 B)."""
    Y = years or par.Y
    W = par.W
    T = W * Y
    ax = grid.axis
    nx = len(ax)
    g = 10.0 ** ax
    S, RA, RB = [a.ravel() for a in np.meshgrid(g, g, g, indexing="ij")]
    V = np.zeros((nx, nx, nx))
    pol = np.zeros((T, nx, nx, nx), dtype=np.int8)
    for t in range(T - 1, -1, -1):
        last = t % W == W - 1
        Q = []
        for a in (0, 1, 2):
            S2, RA2, RB2, c = step2(par, S, RA, RB, a)
            cont = interp3(V, *winter2(par, S2, RA2, RB2), grid) if last else interp3(V, S2, RA2, RB2, grid)
            Q.append(-c + cont)
        Q = np.stack(Q)
        pol[t] = np.argmax(Q, axis=0).reshape(nx, nx, nx).astype(np.int8)
        V = Q.max(axis=0).reshape(nx, nx, nx)
    return V, pol


def value_at2(V, par: Par, grid: Grid = GRID2) -> float:
    S, RA, RB = start_state2(par)
    return 100.0 * par.Y + float(interp3(V, np.array([S]), np.array([RA]), np.array([RB]), grid)[0])


def forward2(par: Par, pol, years=None, grid: Grid = GRID2):
    """Vorwärtslauf der Politik (nächstes Gitterfeld). Rückgabe: Gesamtwert und je Jahr (Spritzungen A, Spritzungen B, Anteil RA, Anteil RB am Saisonende)."""
    Y = years or par.Y
    W = par.W
    S, RA, RB = start_state2(par)
    nx = len(grid.axis)
    total = 0.0
    log = []
    idx = lambda zz: min(max(int(round((np.log10(max(zz, 1e-30)) - grid.lo) / grid.step)), 0), nx - 1)
    for y in range(Y):
        net = 100.0
        acts = []
        for t in range(W):
            a = int(pol[y * W + t, idx(S), idx(RA), idx(RB)])
            S, RA, RB, c = step2(par, S, RA, RB, a)
            net -= c
            acts.append(a)
        total += net
        tot = S + RA + RB
        log.append((acts.count(1), acts.count(2), RA / tot, RB / tot))
        S, RA, RB = winter2(par, S, RA, RB)
    return total, log
