"""Kennzahlen und Verdikte (reine Funktionen; tests/test_verdicts.py prüft jede Schwelle und jede Zahl im Text).

Alle Verdikte sind Aussagen im Modell bei den Annahmen der App, keine Empfehlungen. Rückgabe (Art, Text), Art: success | info | warning.
Schwellen: 5 % und 1 % der Bezugssumme (Kurzsicht, Pausen, Wechsel, Voraussicht), drei Standardfehler des gepaarten Unterschieds (Pausen)."""

import sre_constants as C

BIG_DEN = 20    # „deutlich“: ab 1/20 = 5 % der Bezugssumme (ganzzahliger Nenner statt 0,05: Gleitkomma-Gleichheit an der Grenze)
SMALL_DEN = 100  # „knapp“: ab 1/100 = 1 %
Z = 3.0         # Standardfehler, ab denen ein gepaarter Unterschied nicht mehr im Rauschen liegt


def pct(x, ref):
    """x in Prozent von ref; None, wenn ref 0 ist."""
    return None if ref == 0 else 100.0 * x / ref


def kurzsicht_verdikt(sum_blind, sum_never, sum_horizon):
    """Frage 1: Was kostet es, die Schwelle nur auf das erste Jahr abzustimmen? Bezug ist „nie spritzen“."""
    gap = sum_horizon - sum_blind
    gap_shown = round(sum_horizon) - round(sum_blind)  # aus den angezeigten (gerundeten) Summen, damit die Zahlen im Satz zusammenpassen
    if sum_blind < sum_never:
        return "warning", (
            f"**Kurzsicht kostet mehr als Nichtstun:** die auf 1 Jahr abgestimmte Schwelle erreicht über die Laufzeit {sum_blind:.0f} Ertragspunkte, nie spritzen {sum_never:.0f}; "
            f"die auf den Planungshorizont abgestimmte Schwelle {sum_horizon:.0f} (Unterschied {gap_shown})."
        )
    if gap * BIG_DEN >= sum_never:
        return "info", (
            f"**Kurzsicht kostet spürbar:** die auf 1 Jahr abgestimmte Schwelle liegt {gap_shown} Ertragspunkte unter der auf den Planungshorizont abgestimmten "
            f"({sum_blind:.0f} gegen {sum_horizon:.0f}; nie spritzen {sum_never:.0f})."
        )
    return "success", (
        f"**Kein Gedächtnisproblem:** die auf 1 Jahr und die auf den Planungshorizont abgestimmte Schwelle liegen {abs(gap_shown)} Ertragspunkte auseinander "
        f"({sum_blind:.0f} und {sum_horizon:.0f}; nie spritzen {sum_never:.0f}), unter 5 % von „nie spritzen“."
    )


def pausen_verdikt(d_vs_cal, se, cal_total):
    """Frage 2: Schlagen Pausen (Pausenkalender) den dauernden Kalender? d_vs_cal = Summe Pausenkalender minus Summe Kalender (gepaart), se = Standardfehler dieses Unterschieds."""
    rel = pct(d_vs_cal, cal_total)
    if rel is None:
        return "info", "Die Summe des Kalenders ist 0: ein Vergleich in Prozent ist nicht möglich."
    if abs(d_vs_cal) <= Z * se:
        return "info", f"**Pausen: im Rauschen.** Unterschied zum Kalender {d_vs_cal:+.0f} Ertragspunkte bei einem Standardfehler von {se:.1f} (weniger als drei Standardfehler)."
    if d_vs_cal * BIG_DEN >= cal_total:
        return "success", f"**Pausen lohnen deutlich:** der Pausenkalender liegt {d_vs_cal:.0f} Ertragspunkte ({rel:.1f} %) über dem Kalender."
    if d_vs_cal > 0:
        return "info", f"**Pausen lohnen knapp:** der Pausenkalender liegt {d_vs_cal:.0f} Ertragspunkte ({rel:.1f} %) über dem Kalender."
    return "warning", f"**Pausen schaden:** der Pausenkalender liegt {-d_vs_cal:.0f} Ertragspunkte ({-rel:.1f} %) unter dem Kalender."


def wechsel_verdikt(rotation, best_single, cross):
    """Frage 3: Hilft der Wirkstoffwechsel? (Vergleich nur mit der besten Ein-Wirkstoff-Regel, nicht mit anderen Hebeln.) rotation = Summe der besten Wechselregel, best_single = Summe der besten Regel mit einem Wirkstoff, cross = Kreuzresistenz."""
    d = rotation - best_single
    rel = pct(d, best_single)
    if rel is None:
        return "info", "Die Summe der besten Ein-Wirkstoff-Regel ist 0: ein Vergleich in Prozent ist nicht möglich."
    if d * BIG_DEN >= best_single:
        text = f"**Wirkstoffwechsel hilft deutlich:** die beste Wechselregel liegt {d:.0f} Ertragspunkte ({rel:.1f} %) über der besten Regel mit einem Wirkstoff."
        if cross > 0:
            text += f" Die Kreuzresistenz {cross:g} nimmt davon einen Teil."
        return "success", text
    if d * SMALL_DEN > best_single:
        return "info", f"**Wirkstoffwechsel hilft wenig:** {d:.0f} Ertragspunkte ({rel:.1f} %) über der besten Regel mit einem Wirkstoff" + (f"; Kreuzresistenz {cross:g}." if cross > 0 else ".")
    return "warning", (
        f"**Wirkstoffwechsel bringt hier nichts:** Unterschied {d:+.0f} Ertragspunkte ({rel:+.1f} %) zur besten Regel mit einem Wirkstoff"
        + ("; bei Kreuzresistenz 1 sind beide Wirkstoffe gleich." if cross >= 1.0 else ".")
    )


def voraussicht_verdikt(dp, myopic, best_cal):
    """Frage 4: Was bringt volle Voraussicht? Referenz der deterministischen Instanz (volle Information, kein Rauschen): Rückwärtsrechnung gegen die jahresweise beste Antwort."""
    rel = pct(dp - myopic, myopic)
    if rel is None:
        return "info", "Die Summe der jahresweise besten Antwort ist 0: ein Vergleich in Prozent ist nicht möglich."
    base = f"Referenz ohne Rauschen und mit bekanntem Zustand: Rückwärtsrechnung {dp:.0f}, jahresweise beste Antwort {myopic:.0f}, bester Kalender {best_cal:.0f}."
    if (dp - myopic) * BIG_DEN >= myopic:
        return "info", f"**Voraussicht lohnt:** {rel:.1f} % mehr als die jahresweise beste Antwort. {base}"
    if (dp - myopic) * SMALL_DEN >= myopic:
        return "info", f"**Voraussicht lohnt knapp:** {rel:.1f} % mehr als die jahresweise beste Antwort. {base}"
    return "success", f"**Voraussicht bringt hier kaum etwas:** {rel:.1f} % gegenüber der jahresweise besten Antwort. {base}"


def kipp_verdikt(phi, cost, s1, s20, sum1, sum20):
    """Raster: gilt „die auf 20 Jahre abgestimmte Schwelle liegt weit über der auf 1 Jahr abgestimmten“ (S20 mindestens C.KIPP_S der Kapazität)?"""
    base = f"Nächste Rasterzelle φ = {phi:g}, c = {cost:g}: S(1 Jahr) = {s1:.3f}, S(20 Jahre) = {s20:.3f}; Summen {sum1:.0f} (1 Jahr) und {sum20:.0f} (20 Jahre)."
    if s20 >= C.KIPP_S:
        return "warning", f"**Gedächtnis-Dilemma:** die Voraussicht verlangt eine Schwelle weit über der kurzsichtigen. {base}"
    return "success", f"**Kein Gedächtnisproblem im Raster:** die auf 20 Jahre abgestimmte Schwelle bleibt klein (unter {C.KIPP_S:g}), die Summen liegen {abs(round(sum20) - round(sum1))} Ertragspunkte auseinander. {base}"


def lueckentext(seconds, paths_train, paths_eval, source):
    """Zeile über die Herkunft der Regeltabelle."""
    if source == "vorgerechnet":
        return "Vorgerechnet: abgestimmt auf 400 Trainingspfaden (Seed 11), ausgewertet auf 2000 getrennten Pfaden (Seed 22)."
    return (f"Schnellabstimmung (Näherung, {seconds:.0f} s): abgestimmt auf {paths_train} Trainingspfaden (Seed 11), ausgewertet auf {paths_eval} getrennten Pfaden; "
            "die vorgerechneten Tabellen verwenden 400 und 2000 Pfade.")
