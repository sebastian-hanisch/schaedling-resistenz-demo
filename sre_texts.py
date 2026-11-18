"""Texte, die App und README gemeinsam führen (ein Test vergleicht beide Fassungen)."""

ASSUMPTIONS_MARKDOWN = """
- **Alle Zahlen sind Annahmen:** weder an Schädlingen oder Wirkstoffen gemessen noch aus der Literatur (keine Literaturwerte). Wachstumsrate 0,45 je Woche, Kapazität 1, Schaden gesättigt mit Halbsättigung 0,25 (Holling Typ II, eigene Wahl), Abtötungsrate 0,9, Überwinterung 0,01, Zuwanderung 0,01 mit Resistenzanteil 10⁻⁵ je Wirkstoff, Wetterstreuung 0,15, Streuung der Zuwanderung 0,5. Jede Aussage der App gilt im Modell bei diesen Annahmen und ist keine Spritzempfehlung.
- **Die Standardwerte wurden nach ersten Läufen gesetzt (ein Freiheitsgrad):** die erste Fassung (Resistenzgrad 0,95, Fitnesskosten 0,1, Kontrollkosten 0,3) ließ in der Vormessung (anderer Code-Stand) den Wirkstoff schnell versagen und die Schwellenregeln im Kontrollaufwand untergehen; mit dem heutigen Modell gilt das bei diesen Werten nur für die kurzsichtige Schwelle (Nachmessung, nicht in den Tests); danach wurden Resistenzgrad 0,9, Fitnesskosten 0,3 und Kontrollkosten 0,1 gewählt. Die Raster und Presets zeigen, wie die Befunde von dieser Wahl abhängen; die Wahl selbst bleibt frei.
- **Klonal und haploid:** die vier Genotypen vermehren sich klonal (blattlausartig), ohne Dominanz, ohne Kreuzung und ohne Mutation in den 20 Jahren. Doppelresistente entstehen nur aus der Zuwanderung (Standard: Produkt der Einzelanteile). Deshalb werden Wirkstoffwechsel und Refugium anders beurteilt als bei geschlechtlicher Vermehrung.
- **Refugium (nur die zwei Refugium-Zeilen):** im klonalen Modell schneidet das Refugium in jedem vorgerechneten Szenario schlechter ab als dieselbe Regel ohne Refugium (Zählung in der README), vermutlich, weil die unbehandelte Fläche selbst Ertrag verliert und die Durchmischung im Winter die Resistenz kaum verdünnt (beide Ursachen nicht getrennt gemessen). Das ist ein Modellartefakt dieser Genetik und widerspricht der Praxisregel (rezessive Resistenz, Kreuzung der Überlebenden) nicht; es zeigt nur, dass dieses Modell Refugien nicht abbildet.
- **Frühe Stoß-Spritzung gewinnt im Modell**, vermutlich wegen des exponentiellen Wachstums und der gesättigten Schadensfunktion (Ursache nicht getrennt gemessen); in der Praxis begrenzen Auflagen Zahl und Abstände der Spritzungen. Der Befund hängt an der Schadensform: bei sehr flachem Schaden (Halbsättigung 4) lohnt Spritzen kaum (Vormessung, nicht in der App).
- **Einschwingzeit:** das erste Jahr beginnt mit einem neuen Wirkstoff (Resistenz nur als Hintergrund 10⁻⁵). Das ist ein Szenario, kein Gleichgewicht; die Summen über 20 Jahre enthalten diesen Anfangsvorteil, deshalb stehen Fünfjahresblöcke in der Tabelle. Ein Lauf über 60 Jahre wurde nicht gerechnet.
- **Abstimmung:** jede Regel ist auf Trainingspfaden abgestimmt (Gitter von Schwellen, Kalendern, Obergrenzen) und auf getrennten Pfaden gemessen; die Tabellen sind vorgerechnet (400 Trainings-, 2000 Auswertungspfade). Die Schnellabstimmung in der App ist eine Näherung (weniger Pfade). Die „blind“ auf 1 Jahr abgestimmte Schwelle ist ausdrücklich der Kurzsicht-Fall, kein Strohmann: daneben stehen die auf 5 Jahre und auf den Planungshorizont abgestimmte Schwelle.
- **Die Rückwärtsrechnung (volle Voraussicht) ist nur eine Referenz:** deterministisch (kein Rauschen, Kontrollen gratis und exakt, kein Refugium, keine Abzinsung), ein Wirkstoff auf einem Gitter von 0,05 Dekaden bzw. zwei Wirkstoffe ohne Doppelresistente auf 0,1 Dekaden (Politik am nächsten Gitterfeld). Sie ist deshalb keine Schranke für die verrauschte Instanz. Sie teilt die Wochengleichung mit der Simulation; ihre Richtigkeit prüfen die Orakeltests mit einer Aufzählung aller Spritzpläne einer kleinen Instanz und einer zweiten Fassung in reinem Python, nicht der Vergleich mit der Simulation.
- **Nicht modelliert:** Mutation und Rekombination, stochastische Rückwärtsrechnung, Rückwärtsrechnung mit Doppelresistenz, Zeitverzögerung der Wirkung (Abbau des Wirkstoffs), Wanderung zwischen Flächen, mehr als zwei Wirkstoffe, Mischungen, Spritzauflagen als harte Regel, Lernen der Resistenz aus Messungen, ein Gleichgewicht über 60 Jahre.
- **Fachliche Vorlage (nur Abstract gelesen):** Hueth und Regev (1974), „Optimal Agricultural Pest Management with Increasing Pest Resistance“, American Journal of Agricultural Economics 56(3), 543 bis 552. Nach dem Abstract bleiben die Nutzerkosten der Resistenz unberücksichtigt, wenn Landwirte sie ignorieren, und die Schwelle steigt unter vernünftigen Bedingungen im Saisonverlauf; der Volltext war nicht zugänglich. Diese Demo rechnet die Arbeit nicht nach.
"""

MATH_NOTE = (
    "**Das Modell ist damit vollständig beschrieben.** *Anfangswerte:* im ersten Jahr beginnt jede Fläche mit $n_{g}=m\\,\\pi_g$ (nur Zuwanderer, $\\pi_{SS}=(1-p)^2$, $\\pi_{RS}=\\pi_{SR}=p(1-p)$, $\\pi_{RR}=p^2$ mit $p=10^{-5}$; "
    "mit dem Regler für Doppelresistenz $\\pi_{RR}$ vorgegeben und $\\pi_{SS}$ entsprechend verkleinert). "
    "*Reihenfolge einer Woche:* Entscheidung der Regel, Spritzung (vor dem Wachstum), Wachstum, Schaden aus dem Mittel der Gesamtdichte vor und nach dem Wachstum; Kosten der Spritzung $c_s(1-q)$, Kosten einer Kontrolle $c_k$ in jeder Kontrollwoche. "
    "*Jahreswechsel:* am Saisonende bilden behandelte Fläche und Refugium einen gemeinsamen Pool (Anteil $1-q$ und $q$), der Winter setzt die Dichte auf $\\nu\\cdot$Pool plus Zuwanderung. "
    "*Zufall:* je Pfad und Jahr ein Wetterfaktor $\\rho_y=\\exp(\\sigma_w z-\\sigma_w^2/2)$ und eine Zuwanderung $\\mu_y=\\exp(\\sigma_m z'-\\sigma_m^2/2)$ (beide mit Mittelwert 1, $z,z'$ standardnormal), je Kontrollwoche ein Messfehler $e^{\\sigma_k z''}$ (lognormal); alle Regeln eines Laufs sehen dieselben Zufallszahlen. "
    "*Randfälle:* der Kalender mit $k=1000$ spritzt nie; die Schwelle prüft nur in Wochen mit $t \\bmod v=0$ (Woche 0 ist Kontrollwoche); die Obergrenze zählt die Spritzungen der laufenden Saison; „nur jedes $\\text{per}$-te Jahr“ meint die Jahre mit Index $y \\bmod \\text{per}=0$, das erste Jahr hat Index 0; "
    "Wechsel je Jahr beginnt mit A, Wechsel je Spritzung zählt die Spritzungen über alle Jahre; ohne Pool ($m$ klein, keine Überlebenden) bliebe der Resistenzanteil unbestimmt, deshalb ist $m\\ge 0{,}001$. "
    "Die Rückwärtsrechnung verwendet dieselbe Wochengleichung ohne Zufall (alle Zufallsfaktoren 1) mit exakter Messung."
)

METHODS_MARKDOWN = """
| Regel | Entscheidung | Abgestimmte Größen |
|---|---|---|
| Nie spritzen | keine Spritzung | keine |
| Kalender | spritze in den Wochen $w_0, w_0+k, \\dots$ jeder Saison | Abstand $k$, Start $w_0$ (Gitter: $k$ = 1 bis 13, 16, 20, 26, Start 0 bis 7) |
| Schwelle | spritze, wenn die gemessene Dichte $S$ überschreitet (lognormaler Messfehler, Kosten je Kontrolle) | $S$ (36 Werte von 0,003 bis 1,5), Kontrollabstand $v$ = 1, 2 oder 4 Wochen |
| Schwelle, nur auf 1 Jahr abgestimmt | wie Schwelle, aber $S$ maximiert nur das erste Jahr (Kurzsicht) | $S$, $v$ |
| Schwelle, auf 5 Jahre / Planungshorizont | $S$ maximiert die ersten 5 Jahre bzw. $H$ Jahre | $S$, $v$ |
| Schwelle mit Obergrenze | Schwelle, aber höchstens `cap` Spritzungen je Saison | $S$, $v$, cap = 1, 2, 3, 4, 6 |
| Pausenkalender | Stoß: höchstens `cap` Spritzungen im Wochenabstand $k$ ab Woche $w_0$, nur in jedem `per`-ten Jahr | $k$ = 1, 2, 4; $w_0$ = 0, 2, 4; cap = 1 bis 5; per = 1, 2, 3, 4, 6 (225 Kombinationen) |
| Wechsel A/B | Kalender oder Schwelle, abwechselnd Wirkstoff A und B, je Jahr oder je Spritzung | wie die Regel ohne Wechsel |
| A dann B | Schwelle mit Wirkstoff A bis zu einem abgestimmten Jahr, danach B | $S$, Wechseljahr |
| Refugium | Schwelle (auch mit Wechsel je Jahr) auf dem Flächenanteil $1-q$, der Anteil $q$ wird nie gespritzt | $S$, $v$ |
| Rückwärtsrechnung (Referenz) | exakte dynamische Optimierung auf der deterministischen Instanz, Entscheidung je Woche | keine (Gitter 0,05 Dekaden) |

Jede Regel wird auf Trainingspfaden (Zufalls-Seed 11) abgestimmt und auf getrennten Auswertungspfaden gemessen, alle Regeln mit denselben Zufallszahlen (gepaarte Unterschiede, Standardfehler des Unterschieds in der Tabelle). Die vorgerechneten Tabellen verwenden 400 Trainings- und 2000 Auswertungspfade, die Schnellabstimmung 60 und die gewählte Pfadzahl.
"""
