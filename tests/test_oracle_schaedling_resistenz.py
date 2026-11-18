"""Orakel-Tests (ORACLE-PLAYBOOK): unabhängige Rechenwege für den Kern.

(A)  Aufzählung aller Spritzpläne einer Mini-Instanz (8 Wochen, 2 Jahre, 2^16 Pläne, reines Python) gegen die Rückwärtsrechnung (Gitterfehler als Band).
(A2) Zwei Wirkstoffe: Aufzählung aller 3^8 Pläne (4 Wochen, 2 Jahre) gegen die Rückwärtsrechnung mit drei Zuständen.
(A3) Zweite Fassung der Rückwärtsrechnung in reinem Python ohne `map_coordinates` auf einem groben Gitter: gleiche Werte wie die Demo.
(B)  Auslese: Anteil Resistenter nach k Spritzungen = p·sR^k / (p·sR^k + (1 − p)·sS^k) in exakten Brüchen, gegen die Simulation ohne Wachstum.
(C)  Handrechnung einer Woche (Python-`math`, Brüche für die Spritzung) gegen `ricker_week`, `week_step` und die Simulation.
(D)  Zeichenweise Python-Schleife (ein Pfad) gegen die NumPy-Simulation mit denselben Zufallszahlen: alle Regeln, alle Wirkstoffwahlen, Refugium, Kreuz- und Doppelresistenz.
(E)  Grenzfälle mit bekannter Lösung: ohne Wachstum (Brüche), ohne Schaden, φ = 0, Refugium 1, Abtötungsrate 0, Kreuzresistenz 1.
(F)  Rückwärtsrechnungs-Hilfsfunktionen (Schwelle, Kalender) gegen die Simulation ohne Zufall.
(G)  Abstimmung: das Gitter wird von Hand durchlaufen, `tune` wählt den besten Punkt auf den Trainingspfaden.
"""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import sre_dp as DP
import sre_hand as H
import sre_model as MD
import sre_rules as R
from sre_model import Par
from sre_scenario import DEFAULTS, par_from_settings

from conftest import mini


# ================================================================================================ (A) Aufzählung, ein Wirkstoff
def _enumerate_best(par):
    """Bester Spritzplan aller 2^(W·Y) Pläne (Tiefensuche mit gemeinsamen Anfangsstücken), reines Python, deterministische Instanz."""
    W, Y = par.W, par.Y
    sus, res = 1 - par.kill, 1 - par.kill * (1 - par.phi)
    fR = 1 - par.cost_fit
    best = [-1e18, None]

    def rec(week, S, R, value, plan):
        if week == W * Y:
            if value > best[0]:
                best[0], best[1] = value, tuple(plan)
            return
        for a in (0, 1):
            s, r = (S * sus, R * res) if a else (S, R)
            N1 = s + r
            g = 1 - N1 / par.K
            s2, r2 = s * math.exp(par.r * g), r * math.exp(par.r * fR * g)
            Nm = 0.5 * (N1 + s2 + r2)
            v = value - par.b * 100.0 / W * Nm / (Nm + par.h) - par.c_spray * a
            if week % W == W - 1:  # Jahreswechsel
                s2, r2 = par.nu * s2 + par.m * (1 - par.p0), par.nu * r2 + par.m * par.p0
            plan.append(a)
            rec(week + 1, s2, r2, v, plan)
            plan.pop()

    rec(0, par.m * (1 - par.p0), par.m * par.p0, 100.0 * Y, [])
    return best[0], best[1]


@pytest.fixture(scope="module")
def mini_par():
    return DP.det_par(mini(phi=0.9, cost_fit=0.3, m=0.01, p0=1e-3, c_spray=1.0))


@pytest.fixture(scope="module")
def mini_enum(mini_par):
    return _enumerate_best(mini_par)


def test_a_aufzaehlung_gleich_rueckwaertsrechnung(mini_par, mini_enum):
    best_value, best_plan = mini_enum
    V, diffs = DP.solve(mini_par)
    v_back = DP.value_at(V, mini_par)
    fwd, sprays, res, weeks = DP.forward(mini_par, diffs)
    assert len(best_plan) == 16
    assert fwd <= best_value + 1e-9  # eine Politik der Rückwärtsrechnung ist ein Plan: nie besser als der beste Plan
    assert best_value - fwd < 0.01  # Gitterfehler (0,05 Dekaden, bilinear) als Band: hier findet die Politik sogar genau den besten Plan
    assert abs(v_back - best_value) < 0.05
    # die Referenz ist keine Schranke nach oben für den Wert auf dem Gitter, aber die Aufzählung ist es für jede Politik
    assert best_value > DP.calendar_value(mini_par, 1000, 1000) + 1.0  # Spritzen lohnt in dieser Instanz


def test_a_aufzaehlung_findet_den_besten_kalender_nicht_besser_als_den_besten_plan(mini_par, mini_enum):
    best_value, _ = mini_enum
    cal_best = max(DP.calendar_value(mini_par, k, w0) for k in range(1, 9) for w0 in range(0, min(k, 8)))
    assert cal_best <= best_value + 1e-9


def test_a_aufzaehlung_stimmt_mit_der_vorwaertsrechnung_der_pläne_ueberein(mini_par, mini_enum):
    """Der beste Plan der Aufzählung, vorwärts mit den Hilfsfunktionen der Demo gerechnet, ergibt denselben Wert (zwei Rechenwege für dieselbe Zahl)."""
    best_value, plan = mini_enum
    S, R_ = DP.start_state(mini_par)
    total = 0.0
    for y in range(mini_par.Y):
        net = 100.0
        for t in range(mini_par.W):
            S, R_, c = DP.week_step(mini_par, S, R_, plan[y * mini_par.W + t])
            net -= c
        total += net
        S, R_ = DP.winter(mini_par, S, R_)
    assert total == pytest.approx(best_value, abs=1e-9)


# ================================================================================================ (A2) Aufzählung, zwei Wirkstoffe
def test_a2_zwei_wirkstoffe_aufzaehlung_gegen_rueckwaertsrechnung():
    par = DP.det_par2(mini(W=4, Y=2, phi=0.9, cost_fit=0.3, m=0.01, p0=1e-3, c_spray=1.0))
    sv = MD.survival(par)
    fR = 1 - par.cost_fit
    pa = pb = par.p0
    best = [-1e18]

    def rec(week, S, RA, RB, value):
        if week == par.W * par.Y:
            best[0] = max(best[0], value)
            return
        for a in (0, 1, 2):
            s, ra, rb = S, RA, RB
            if a:
                s, ra, rb = S * sv[a - 1][0], RA * sv[a - 1][1], RB * sv[a - 1][2]
            N1 = s + ra + rb
            g = 1 - N1 / par.K
            s2, ra2, rb2 = s * math.exp(par.r * g), ra * math.exp(par.r * fR * g), rb * math.exp(par.r * fR * g)
            Nm = 0.5 * (N1 + s2 + ra2 + rb2)
            v = value - 100.0 / par.W * Nm / (Nm + par.h) - (par.c_spray if a else 0.0)
            if week % par.W == par.W - 1:
                s2, ra2, rb2 = par.nu * s2 + par.m * (1 - pa - pb), par.nu * ra2 + par.m * pa, par.nu * rb2 + par.m * pb
            rec(week + 1, s2, ra2, rb2, v)

    rec(0, par.m * (1 - pa - pb), par.m * pa, par.m * pb, 100.0 * par.Y)
    grid = DP.Grid(-7.0, 0.6, 0.25)
    V, pol = DP.solve2(par, grid=grid)
    v_back = DP.value_at2(V, par, grid)
    fwd, log = DP.forward2(par, pol, grid=grid)
    assert fwd <= best[0] + 1e-9
    assert best[0] - fwd < 1.5 and abs(v_back - best[0]) < 1.5  # grobes Gitter (0,25 Dekaden): Band


# ================================================================================================ (A3) zweite Fassung der Rückwärtsrechnung
def _bilinear(V, x, y, n):
    """Bilinear mit Randfortsetzung (nächstes Randfeld), reines Python."""
    x = min(max(x, 0.0), n - 1.0)
    y = min(max(y, 0.0), n - 1.0)
    i, j = min(int(math.floor(x)), n - 2), min(int(math.floor(y)), n - 2)
    fx, fy = x - i, y - j
    return V[i][j] * (1 - fx) * (1 - fy) + V[i + 1][j] * fx * (1 - fy) + V[i][j + 1] * (1 - fx) * fy + V[i + 1][j + 1] * fx * fy


def _solve_python(par, grid):
    ax = [grid.lo + k * grid.step for k in range(len(grid.axis))]
    n = len(ax)
    W, Y = par.W, par.Y
    V = [[0.0] * n for _ in range(n)]
    diffs = [None] * (W * Y)
    for t in range(W * Y - 1, -1, -1):
        last = t % W == W - 1
        newV = [[0.0] * n for _ in range(n)]
        diff = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                S, R_ = 10.0 ** ax[i], 10.0 ** ax[j]
                q = []
                for a in (0, 1):
                    s, r = (S * (1 - par.kill), R_ * (1 - par.kill * (1 - par.phi))) if a else (S, R_)
                    N1 = s + r
                    g = 1 - N1 / par.K
                    s2, r2 = s * math.exp(par.r * g), r * math.exp(par.r * (1 - par.cost_fit) * g)
                    Nm = 0.5 * (N1 + s2 + r2)
                    cost = 100.0 * par.b / W * Nm / (Nm + par.h) + par.c_spray * a
                    if last:
                        s2, r2 = par.nu * s2 + par.m * (1 - par.p0), par.nu * r2 + par.m * par.p0
                    x = (math.log10(max(s2, 1e-30)) - grid.lo) / grid.step
                    y = (math.log10(max(r2, 1e-30)) - grid.lo) / grid.step
                    q.append(-cost + _bilinear(V, x, y, n))
                diff[i][j] = q[1] - q[0]
                newV[i][j] = max(q)
        V, diffs[t] = newV, diff
    return V, diffs


def test_a3_zweite_fassung_in_reinem_python_gleicht_der_demo():
    par = DP.det_par(mini(W=4, Y=2, phi=0.9, cost_fit=0.3, m=0.01, p0=1e-3, c_spray=1.0))
    grid = DP.Grid(-6.0, 0.5, 0.25)
    V_np, d_np = DP.solve(par, grid=grid)
    V_py, d_py = _solve_python(par, grid)
    assert np.allclose(V_np, np.array(V_py), atol=1e-8)
    assert np.allclose(d_np[0], np.array(d_py[0]), atol=1e-4) and np.allclose(d_np[5], np.array(d_py[5]), atol=1e-4)  # diffs sind float32


# ================================================================================================ (B) Auslese in Brüchen
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_b_auslese_formel_gegen_simulation_ohne_wachstum(k):
    par = Par(W=k, Y=1, r=0.0, b=0.0, kill=0.9, phi=0.9, p0=1e-5, pB0=0.0, rr_null=True)
    res = MD.simulate(par, "cal", dict(k=np.array([1]), w0=np.array([0])), 1, 0, noise=False)
    p = F(1, 100000)
    exact = H.share_after(p, k, F(1, 10), F(91, 100))
    assert res["resA"][0, 0, 0] == pytest.approx(float(exact), rel=1e-12)
    assert exact == (p * F(91, 100) ** k) / (p * F(91, 100) ** k + (1 - p) * F(1, 10) ** k)


def test_b_handbeispiel_zahlen():
    assert H.share_start() == F(1, 101)
    assert H.share_after(H.share_start(), 1, F(1, 10), F(91, 100)) == F(91, 1091)  # von Hand: (1/101 · 91/100) / (1/101 · 91/100 + 100/101 · 1/10) = 91 / (91 + 1000)
    surv = H.survival_exact(cross=F(1, 2))
    assert surv["A"]["SR"] == F(101, 200) and surv["B"]["RS"] == F(101, 200) and surv["A"]["RS"] == F(91, 100)  # 1/10 + 1/2 · (91/100 − 1/10) = 101/200


# ================================================================================================ (C) eine Woche von Hand
@pytest.mark.parametrize("sprayed", [False, True])
def test_c_handrechnung_einer_woche(sprayed):
    S, R_ = 0.01, 1e-7
    k = 0.9
    s1 = S * (1 - k) if sprayed else S
    r1 = R_ * (1 - k * (1 - 0.9)) if sprayed else R_
    N1 = s1 + r1
    s2 = s1 * math.exp(0.45 * (1 - N1))
    r2 = r1 * math.exp(0.45 * 0.7 * (1 - N1))
    Nm = 0.5 * (N1 + s2 + r2)
    loss = 100.0 / 26 * Nm / (Nm + 0.25)
    par = DP.det_par(par_from_settings(DEFAULTS))
    S2, R2, cost = DP.week_step(par, S, R_, 1 if sprayed else 0)
    assert S2 == pytest.approx(s2, rel=1e-13) and R2 == pytest.approx(r2, rel=1e-13) and cost == pytest.approx(loss + (4.0 if sprayed else 0.0), rel=1e-13)
    wk = H.ricker_week(S, R_, sprayed)
    assert wk["S2"] == pytest.approx(s2, rel=1e-13) and wk["R2"] == pytest.approx(r2, rel=1e-13) and wk["loss"] == pytest.approx(loss, rel=1e-13)
    # Simulation mit einer Woche je Saison und einem Jahr: Ertrag = 100·(1 − Wochenschaden mit W = 1) − Kosten
    p1 = Par(W=1, Y=1, m=S + R_, p0=R_ / (S + R_), pB0=0.0, rr_null=True)
    sim = MD.simulate(p1, "cal", dict(k=np.array([1 if sprayed else 1000]), w0=np.array([0 if sprayed else 1000])), 1, 0, noise=False)
    expected = 100.0 - 100.0 * Nm / (Nm + 0.25) - (4.0 if sprayed else 0.0)
    assert sim["net"][0, 0, 0] == pytest.approx(expected, rel=1e-11)


def test_c_hand_saison_in_brüchen_gegen_aufzaehlung_der_formel():
    """Eine Saison ohne Wachstum, Spritzung in Woche 2: jede Zahl als Bruch von Hand."""
    rows, net = H.exact_season({2})
    S2, R2 = F(1, 100) * F(1, 10), F(1, 10000) * F(91, 100)  # nach der Spritzung in Woche 2
    N_before, N_after = F(1, 100) + F(1, 10000), S2 + R2
    h = F(1, 4)
    loss_w1 = F(25) * N_before / (N_before + h)
    loss_w2 = F(25) * N_after / (N_after + h)
    expected = 100 - loss_w1 - 3 * loss_w2 - 4  # Wochen 2, 3, 4 mit der niedrigen Dichte
    assert net == expected
    assert rows[0]["loss"] == loss_w1 and rows[1]["loss"] == loss_w2 and rows[3]["S"] == S2 and rows[3]["R"] == R2


# ================================================================================================ (D) Python-Schleife gegen NumPy
def _ref_path(par, kind, prm, mode, v, z, rho, mig, years):
    """Ein Pfad in reinem Python (Listen, `math`), gleiche Zufallszahlen wie die NumPy-Fassung."""
    W, q = par.W, par.q
    sus, res = 1 - par.kill, 1 - par.kill * (1 - par.phi)
    x = sus + par.cross * (res - sus)
    surv = [[sus, res, x, res], [sus, x, res, res]]
    cc = par.cost_fit
    fit = [1.0, 1 - cc, 1 - cc, (1 - cc) ** 2]
    p = par.p0
    pb = p if par.pB0 < 0 else par.pB0
    if par.rr_null:
        imm = [1 - p - pb, p, pb, 0.0]
    elif par.rr_freq >= 0:
        imm = [1 - p - pb - par.rr_freq, p, pb, par.rr_freq]
    else:
        imm = [(1 - p) * (1 - pb), p * (1 - pb), (1 - p) * pb, p * pb]
    areas = [[par.m * i for i in imm] for _ in range(2)]
    count = 0
    out = dict(net=[], sprays=[], resA=[], resB=[])
    for y in range(years):
        loss = cost = 0.0
        nsp = 0
        for t in range(W):
            Nt = sum(areas[0])
            if kind == "thr":
                if t % v == 0:
                    nhat = Nt * math.exp(par.sig_meas * z[y][t])
                    cost += par.c_scout
                    spray = nhat > prm["S"]
                else:
                    spray = False
            else:
                spray = t >= prm["w0"] and (t - prm["w0"]) % prm["k"] == 0
            if "cap" in prm:
                spray = spray and nsp < prm["cap"]
            if "per" in prm:
                spray = spray and y % prm["per"] == 0
            ing = {"A": 0, "year": y % 2, "spray": count % 2, "seq": int(y >= prm.get("Ysw", 0))}[mode]
            if spray:
                areas[0] = [n * s for n, s in zip(areas[0], surv[ing])]
                cost += par.c_spray * (1 - q)
                nsp += 1
                count += 1
            for a in (0, 1):
                N1 = sum(areas[a])
                g = 1 - N1 / par.K
                areas[a] = [n * math.exp(par.r * rho[y] * f * g) for n, f in zip(areas[a], fit)]
                N2 = sum(areas[a])
                Nm = 0.5 * (N1 + N2)
                loss += par.b / W * Nm / (Nm + par.h) * ((1 - q) if a == 0 else q)
        pool = [(1 - q) * a0 + q * a1 for a0, a1 in zip(areas[0], areas[1])]
        tot = sum(pool)
        out["net"].append(100 * (1 - loss) - cost)
        out["sprays"].append(nsp)
        out["resA"].append((pool[1] + pool[3]) / tot)
        out["resB"].append((pool[2] + pool[3]) / tot)
        n0 = [par.nu * pl + par.m * mig[y] * i for pl, i in zip(pool, imm)]
        areas = [list(n0), list(n0)]
    return out


CASES = [
    ("Kalender A", dict(), "cal", dict(k=20, w0=4), "A", 1),
    ("Kalender Wechsel je Jahr, Kreuzresistenz", dict(cross=0.5), "cal", dict(k=5, w0=1), "year", 1),
    ("Kalender Wechsel je Spritzung", dict(), "cal", dict(k=3, w0=2), "spray", 1),
    ("Schwelle A, Kontrolle alle 2 Wochen", dict(), "thr", dict(S=0.007), "A", 2),
    ("Schwelle Wechsel je Jahr, Refugium", dict(q=0.2), "thr", dict(S=0.05), "year", 4),
    ("Schwelle mit Obergrenze", dict(), "thr", dict(S=0.003, cap=2), "A", 4),
    ("Pausenkalender", dict(), "cal", dict(k=1, w0=0, cap=3, per=2), "A", 1),
    ("Schwelle A dann B", dict(), "thr", dict(S=0.02, Ysw=1), "seq", 2),
    ("Doppelresistenz in der Zuwanderung", dict(rr_freq=1e-4), "cal", dict(k=5, w0=1), "year", 1),
    ("hoher Resistenzgrad, wenig Fitnesskosten", dict(phi=1.0, cost_fit=0.1, q=0.1), "thr", dict(S=0.01), "A", 1),
]


@pytest.mark.parametrize("label, kw, kind, prm, mode, v", CASES, ids=[c[0] for c in CASES])
def test_d_python_schleife_gleich_numpy(label, kw, kind, prm, mode, v):
    par = par_from_settings(DEFAULTS)
    par = MD.with_(par, **kw)
    years, paths = 3, 3
    z, rho, mig = MD.draws(par, paths, 5, years)
    res = MD.simulate(par, kind, {k: np.array([val]) for k, val in prm.items()}, paths, 5, years=years, mode=mode, v=v)
    for pth in range(paths):
        ref = _ref_path(par, kind, prm, mode, v, z[pth], rho[pth], mig[pth], years)
        for key in ("net", "sprays", "resA", "resB"):
            assert res[key][0, pth] == pytest.approx(ref[key], rel=1e-10, abs=1e-10), (label, key, pth)


# ================================================================================================ (E) Grenzfälle mit bekannter Lösung
def test_e_ohne_wachstum_und_ohne_spritzen_der_ertrag_ist_ein_bruch():
    """r = 0, nie spritzen, kein Zufall: Dichte im Jahr y ist m·(1 − ν^y)/(1 − ν), Ertrag 100·(1 − N/(N + h)) in jedem Jahr."""
    par = Par(Y=5, r=0.0)
    res = MD.simulate(par, "cal", MD.never(), 1, 0, noise=False)
    m, nu, h = F(1, 100), F(1, 100), F(1, 4)
    for y in range(1, 6):
        N = m * (1 - nu ** y) / (1 - nu)
        assert res["net"][0, 0, y - 1] == pytest.approx(float(100 * (1 - N / (N + h))), rel=1e-12)
    assert F(100 * 25, 26) == 100 * (1 - F(1, 100) / (F(1, 100) + h))  # Jahr 1: 100 · 25/26


def test_e_ohne_schaden_ist_der_ertrag_100_minus_kosten():
    par = Par(b=0.0, Y=2)
    res = MD.simulate(par, "thr", dict(S=np.array([0.005])), 20, 3, v=4)
    checks = math.ceil(par.W / 4)  # Wochen 0, 4, ..., 24
    assert checks == 7
    expected = 100.0 - par.c_spray * res["sprays"][0] - par.c_scout * checks
    assert res["net"][0] == pytest.approx(expected, abs=1e-9)
    assert res["sprays"][0].sum() > 0  # die Schwelle löst auf diesen Pfaden aus: der Test prüft die Kosten wirklich


def test_e_phi_null_und_keine_fitnesskosten_resistente_sind_anfaellige():
    kw = dict(phi=0.0, cost_fit=0.0)
    a = MD.simulate(Par(p0=1e-3, **kw), "cal", dict(k=np.array([5]), w0=np.array([1])), 5, 2, years=4)
    b = MD.simulate(Par(p0=0.0, **kw), "cal", dict(k=np.array([5]), w0=np.array([1])), 5, 2, years=4)
    assert np.allclose(a["net"], b["net"], atol=1e-12, rtol=1e-12)
    rot = MD.simulate(Par(p0=1e-3, **kw), "cal", dict(k=np.array([5]), w0=np.array([1])), 5, 2, years=4, mode="year")
    assert np.allclose(a["net"], rot["net"], atol=1e-12, rtol=1e-12)  # ohne Resistenzvorteil ist Wechseln wirkungslos


def test_e_volles_refugium_wird_nie_behandelt():
    par = Par(q=1.0, Y=3)
    spray = MD.simulate(par, "cal", dict(k=np.array([2]), w0=np.array([0])), 10, 1)
    nie = MD.simulate(par, "cal", MD.never(), 10, 1)
    assert np.allclose(spray["net"], nie["net"], atol=1e-12)  # die Spritzung trifft nur die Fläche mit Gewicht 0, Kosten (1 − q)·c = 0


def test_e_abtoetungsrate_null_spritzen_kostet_nur():
    par = Par(kill=0.0, Y=3)
    spray = MD.simulate(par, "cal", dict(k=np.array([13]), w0=np.array([0])), 10, 1)
    nie = MD.simulate(par, "cal", MD.never(), 10, 1)
    assert np.allclose(spray["net"], nie["net"] - par.c_spray * spray["sprays"], atol=1e-10)


def test_e_kreuzresistenz_eins_wechseln_ist_wirkungslos():
    par = Par(cross=1.0, Y=4)
    prm = dict(k=np.array([5]), w0=np.array([1]))
    a = MD.simulate(par, "cal", prm, 10, 4)
    rot = MD.simulate(par, "cal", prm, 10, 4, mode="year")
    spr = MD.simulate(par, "cal", prm, 10, 4, mode="spray")
    assert np.allclose(a["net"], rot["net"], atol=1e-12) and np.allclose(a["net"], spr["net"], atol=1e-12)


# ================================================================================================ (F) Hilfsfunktionen der Rückwärtsrechnung gegen die Simulation
@pytest.mark.parametrize("S", [0.01, 0.1, 0.5, 0.9])
def test_f_schwelle_simulation_gleich_hilfsfunktion(S):
    pd = DP.det_par(par_from_settings(DEFAULTS))
    res = MD.simulate(pd, "thr", dict(S=np.array([S])), 2, 1, noise=False, v=1)
    assert MD.total(res, pd)[0, 0] == pytest.approx(DP.threshold_value(pd, S), abs=1e-9)


@pytest.mark.parametrize("k, w0", [(20, 4), (5, 1), (1, 0), (24, 0)])
def test_f_kalender_simulation_gleich_hilfsfunktion(k, w0):
    pd = DP.det_par(par_from_settings(DEFAULTS))
    res = MD.simulate(pd, "cal", dict(k=np.array([k]), w0=np.array([w0])), 2, 1, noise=False)
    assert MD.total(res, pd)[0, 0] == pytest.approx(DP.calendar_value(pd, k, w0), abs=1e-9)


# ================================================================================================ (G) Abstimmung von Hand
def test_g_tune_waehlt_den_besten_gitterpunkt_der_trainingspfade():
    par = Par(Y=3)
    grid = dict(S=R.s_grid(6))
    best, value = R.tune(par, "thr", v=2, grid=grid, P=15, seed=11)
    vals = []
    for s in grid["S"]:
        res = MD.simulate(par, "thr", dict(S=np.array([s])), 15, 11, v=2)
        vals.append(MD.total(res, par)[0].mean())
    assert best["S"][0] == grid["S"][int(np.argmax(vals))] and value == pytest.approx(max(vals), rel=1e-12)
    # getrennte Auswertungspfade: anderer Seed, anderer Wert
    res_eval = MD.simulate(par, "thr", best, 15, 22, v=2)
    assert MD.total(res_eval, par)[0].mean() != pytest.approx(value, abs=1e-6)
