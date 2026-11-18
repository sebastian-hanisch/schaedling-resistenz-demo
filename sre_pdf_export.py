"""PDF-Export (fpdf2, Kernschriften: Umlaute gehen, andere Sonderzeichen werden ersetzt)."""

from fpdf import FPDF

_REPLACE = {"–": "-", "—": "-", "≥": ">=", "≤": "<=", "→": "->", "×": "x", "·": "*", "„": '"', "“": '"', "’": "'", "≈": "~", "−": "-", "φ": "phi", "ν": "nu", "ρ": "rho"}


def _latin(text):
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings, rows, source_note, verdicts):
    """settings: dict {Name: Wert}; rows: Liste {rule, total, se, d_nie, d_cal, params} oder leer; source_note: Herkunft der Tabelle; verdicts: Liste von Sätzen."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Schädlingsbekämpfung: Wirksamkeit ist ein Vorrat"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _latin("Modellrechnung mit frei gewählten Zahlen (keine Messungen an Schädlingen oder Wirkstoffen, keine Literaturwerte). Keine Spritzempfehlung."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Einstellungen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _latin("; ".join(f"{k} = {v}" for k, v in settings.items())), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Regeln (Summe der Ertragspunkte über 20 Jahre)", new_x="LMARGIN", new_y="NEXT")
    if rows:
        pdf.set_font("Helvetica", "B", 8)
        for head, wd in (("Regel", 92), ("Summe", 22), ("SE", 14), ("gegen nie", 24), ("gegen Kal.", 24)):
            pdf.cell(wd, 5, head, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for r in rows:
            pdf.cell(92, 5, _latin(r["rule"])[:62], border=1)
            pdf.cell(22, 5, f"{r['total']:.0f}", border=1)
            pdf.cell(14, 5, f"{r['se']:.1f}", border=1)
            pdf.cell(24, 5, "" if r.get("d_nie") is None else f"{r['d_nie']:+.0f}", border=1)
            pdf.cell(24, 5, "" if r.get("d_cal") is None else f"{r['d_cal']:+.0f}", border=1)
            pdf.ln()
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, _latin(source_note), new_x="LMARGIN", new_y="NEXT")
    for v in verdicts:
        pdf.ln(1)
        pdf.multi_cell(0, 5, _latin(v.replace("**", "")), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
