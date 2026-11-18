"""Schädlingsbekämpfung: Wirksamkeit ist ein Vorrat - interaktive Fall-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Schwelle, Kalender, Pausen und Wirkstoffwechsel über 20 Jahre mit Resistenz als Gedächtnis; Kurzsicht gegen Voraussicht, Rückwärtsrechnung als Referenz, vorgerechnetes
Raster Resistenzgrad x Fitnesskosten. Alle Zahlen sind Annahmen (README, „Ehrliche Grenzen“); keine Spritzempfehlung.
Lauffähig mit: streamlit run app.py
"""

import dataclasses
import random
import time

import numpy as np
import streamlit as st

import sre_constants as C
import sre_evaluation as EV
import sre_grid as G
import sre_hand as H
import sre_model as MD
import sre_rules as R
import sre_texts as TX
import sre_visualization as V
from sre_layout import dunkles_design, kopf_css
from sre_pdf_export import build_pdf
from sre_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    STATE_OF,
    settings_from_state,
    sidebar_presets,
    sync_query_params,
)
from sre_scenario import DEFAULTS, par_from_settings, table_key

st.set_page_config(page_title="Schädlingsbekämpfung: Wirksamkeit ist ein Vorrat – Sebastian Hanisch", layout="wide")


# ----------------------------------------------------------------------------------------------------------- vorgerechnete Daten
@st.cache_data(show_spinner=False)
def _pre():
    return G.load_regeln()


@st.cache_data(show_spinner=False)
def _raster():
    return G.load_raster()


@st.cache_data(show_spinner=False)
def _horizont():
    return G.load_horizont()


@st.cache_data(show_spinner=False)
def _dp():
    return G.load_dp_werte(), G.load_dp_politik(), G.load_dp2_politik()


# ----------------------------------------------------------------------------------------------------------- Rechnungen der App
@st.cache_data(show_spinner=False, max_entries=6)
def _quick(items, p_train, paths, seed):
    """Schnellabstimmung aller Regeln für die Einstellungen (Näherung): Rückgabe (Zeilen, Sekunden)."""
    s = dict(items)
    par = par_from_settings(s)
    t0 = time.time()
    rules = R.tune_rules(par, int(s["horizon"]), p_train, q_ref=float(s["q"]))
    rows = R.rule_table(par, rules, paths, seed)
    return rows, time.time() - t0


@st.cache_data(show_spinner=False, max_entries=12)
def _fallback(items, grundfall_specs, paths, seed):
    """Die Regeln des Grundfalls (nicht neu abgestimmt) bei den aktuellen Einstellungen auswerten."""
    s = dict(items)
    par = par_from_settings(s)
    rules = []
    for name, spec in grundfall_specs:
        rule = R.Rule.from_json(name, spec)
        rules.append(dataclasses.replace(rule, q=float(s["q"])) if rule.q > 0 else rule)
    return R.rule_table(par, rules, paths, seed)


@st.cache_data(show_spinner=False, max_entries=40)
def _curves(items, name, spec, paths, seed):
    """Ertrag und Resistenzanteil je Jahr einer Regel auf den Auswertungspfaden."""
    par = par_from_settings(dict(items))
    rule = R.Rule.from_json(name, spec)
    tot, res = R.evaluate_rule(par, rule, paths, seed)
    net = res["net"][0]
    return dict(mean=net.mean(0), lo=np.quantile(net, 0.1, axis=0), hi=np.quantile(net, 0.9, axis=0), resA=res["resA"][0].mean(0), resB=res["resB"][0].mean(0), mode=rule.mode)


@st.cache_data(show_spinner=False, max_entries=40)
def _season(items, name, spec, year):
    """Eine Saison (Jahr `year`) einer Regel ohne Zufall: Dichten je Woche, gespritzte Wochen, Wochenschäden."""
    par = par_from_settings(dict(items))
    rule = R.Rule.from_json(name, spec)
    pe = MD.with_(par, q=rule.q)
    res = MD.simulate(pe, rule.kind, rule.arrays(), 1, 0, years=year, mode=rule.mode, noise=False, v=rule.v, record=True)
    sp = res["trace_spray"][0, year - 1]
    return dict(n=res["trace_n"][0, year - 1], weeks=[int(t) for t in np.flatnonzero(sp > 0.5)], loss=res["trace_loss"][0, year - 1], net=float(res["net"][0, 0, year - 1]),
                sprays=int(res["sprays"][0, 0, year - 1]), resA=float(res["resA"][0, 0, year - 1]))


@st.cache_data(show_spinner=False, max_entries=8)
def _profile(items, v, paths, seed):
    return R.profile(par_from_settings(dict(items)), v, paths, seed)


# ----------------------------------------------------------------------------------------------------------- Kopf und Einleitung
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🐛 Schädlingsbekämpfung: Wirksamkeit ist ein Vorrat")
st.markdown(
    """
Wer jedes Jahr so spritzt, wie es **dieses Jahr** am meisten bringt, verbraucht die Wirkung des Mittels: Resistente bleiben übrig und vermehren sich. Die Demo rechnet über **20 Jahre** (26 Wochen je Saison, vier Genotypen, zwei Wirkstoffe) und vergleicht **Kalender**, **Schwelle**, **Pausen** und **Wirkstoffwechsel**. Gefragt wird: Was kostet **Kurzsicht** gegen **Voraussicht**, lohnen **Pausen**, und hilft der **Wirkstoffwechsel**?

Alle Zahlen sind **Annahmen** (keine Messungen, keine Literaturwerte; die Voreinstellungen wurden nach ersten Läufen gesetzt). Die Aussagen gelten im Modell, nicht als Spritzempfehlung. Direkt darunter erklärt der Expander **💬 In einfachen Worten** das Ganze ohne Formeln. Weiter unten: die **Kernfragen**, dann die Expander **🔧 Wie wir das erreichen – vollständiger Methodenvergleich**, **📐 Mathematische Formulierung** und **⚠️ Annahmen, Grenzen und bekannte Probleme**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/schaedling-resistenz-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Wer jedes Jahr so spritzt, wie es **dieses Jahr** am meisten bringt, verbraucht die Wirkung des Spritzmittels: Resistente überleben, vermehren sich und setzen sich durch. Die Anwendung rechnet über 20 Jahre mit vier Sorten Schädlingen (anfällig, resistent gegen Wirkstoff A, gegen B oder gegen beide) und vergleicht **Kalender**, **Schwelle**, **Pausen** und **Wirkstoffwechsel**.

Gefragt wird, was **Kurzsicht** kostet (im Grundfall liegt eine nur auf ein Jahr eingestellte Schwelle unter „nie spritzen“), ob **Pausen** lohnen und ob der **Wirkstoffwechsel** hilft. Alle Zahlen sind Annahmen, weder gemessen noch aus der Literatur; die Voreinstellungen wurden nach ersten Läufen gesetzt. Die Anwendung gibt keine Spritzempfehlung.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/schaedling-resistenz-erklaert.html)
        """
    )
# erklaerseite:einfach:end
st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()


def _new_seed():
    st.session_state["seed_input"] = random.randint(*bounds("seed_input"))


with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Wirkstoff und Resistenz**")
    st.slider("Resistenzgrad φ", *bounds("phi_slider"), key="phi_slider", step=0.05,
              help="Resistente werden nur zum Anteil Abtötungsrate · (1 − φ) getötet: φ = 0 wie anfällig, φ = 1 ganz unempfindlich.")
    st.slider("Fitnesskosten c der Resistenz", *bounds("cost_slider"), key="cost_slider", step=0.05,
              help="Resistente wachsen je Resistenz um den Faktor (1 − c) langsamer (Wachstumsrate mal (1 − c)). Hohe Kosten bremsen die Resistenz von selbst.")
    st.slider("Abtötungsrate der Anfälligen", *bounds("kill_slider"), key="kill_slider", step=0.01, help="Anteil der anfälligen Tiere, den eine Spritzung tötet.")
    st.slider("Kreuzresistenz", *bounds("cross_slider"), key="cross_slider", step=0.05,
              help="Anteil des Resistenzschutzes, der auch gegen den anderen Wirkstoff gilt (0 = keiner; 1 = beide Wirkstoffe wirken wie einer). Wirkt vor allem auf Regeln mit Wirkstoffwechsel; Regeln mit einem Wirkstoff ändern sich nur über die wenigen gegen B Resistenten (bei Kreuzresistenz 1 um 6 bis 16 Punkte).")
    st.slider("Doppelresistente in der Zuwanderung (Zehnerpotenz)", *bounds("rr_slider"), key="rr_slider", format="1e%d",
              help="Anteil Zuwanderer, die gegen beide Wirkstoffe resistent sind. −10 = Produkt der beiden Einzelanteile (10⁻⁵ mal 10⁻⁵). Doppelresistente entstehen im Modell sonst nicht (keine Mutation).")
    st.markdown("**Schädling und Kosten**")
    st.slider("Zuwanderung m je Winter", *bounds("m_slider"), key="m_slider", step=0.001, format="%.3f",
              help="Dichte der Zuwanderer zu Saisonbeginn (Kapazität = 1), mit Resistenzanteil 10⁻⁵ je Wirkstoff. Mindestens 0,001, damit der Pool nie leer ist.")
    st.slider("Kosten einer Spritzung (Ertragspunkte)", *bounds("spray_slider"), key="spray_slider", step=0.5, help="Abzug vom Jahresertrag je Spritzung (ein Ertragspunkt = ein Prozent des Höchstertrags).")
    st.slider("Kosten einer Kontrolle (Ertragspunkte)", *bounds("scout_slider"), key="scout_slider", step=0.05, help="Abzug je Kontrolle; nur Schwellenregeln kontrollieren.")
    st.slider("Messrauschen (lognormale Streuung)", *bounds("meas_slider"), key="meas_slider", step=0.05, help="Streuung der gemessenen Dichte um die wahre Dichte; nur Schwellenregeln messen.")
    st.markdown("**Planung**")
    st.slider("Planungshorizont der Abstimmung (Jahre)", *bounds("horizon_slider"), key="horizon_slider",
              help="Über so viele Jahre werden die Regeln abgestimmt (außer den Zeilen „1 Jahr“ und „5 Jahre“); ausgewertet wird immer über 20 Jahre.")
    st.slider("Zinsfaktor je Jahr", *bounds("disc_slider"), key="disc_slider", step=0.05,
              help="Jedes spätere Jahr zählt mit diesem Faktor weniger (1 = keine Abzinsung); wirkt auf Abstimmung und Auswertung.")
    st.slider("Refugium-Anteil q", *bounds("q_slider"), key="q_slider", step=0.05,
              help="Flächenanteil, der nie gespritzt wird. Gilt nur für die zwei Refugium-Zeilen der Tabelle; alle anderen Regeln laufen ohne Refugium.")
    st.markdown("**Auswertung**")
    st.number_input("Zufalls-Seed der Auswertungspfade", *bounds("seed_input"), key="seed_input", step=1,
                    help="Wirkt auf die Kurven und die Schnellabstimmung der App; die vorgerechneten Tabellen haben den Zufalls-Seed 22.")
    st.slider("Pfadzahl der Auswertung in der App", *bounds("paths_slider"), key="paths_slider", step=50, help="Zahl der Zufallspfade für Kurven, Schnellabstimmung und Profil.")
    st.button("🎲 Neuen Zufalls-Seed würfeln", on_click=_new_seed, key="new_seed")

settings = settings_from_state()
sync_query_params({state: settings[name] for name, state in STATE_OF.items()})
items = tuple(sorted(settings.items()))
par = par_from_settings(settings)
paths, seed = int(settings["paths"]), int(settings["seed"])
disc = float(settings["disc"])
total_label = "Summe über 20 Jahre" if disc >= 1.0 else f"Summe über 20 Jahre (abgezinst, Faktor {disc:g})"

# ----------------------------------------------------------------------------------------------------------- Regeltabelle für die Einstellungen
pre = _pre()
dp_werte, dp_politik, dp2_politik = _dp()
match = next((name for name, sc in pre.items() if table_key(sc["settings"]) == table_key(settings)), None)
quick_key = table_key(settings) + (paths, seed)
rows, source, source_text = [], None, ""
tuned = False
if match is not None:
    rows, source, tuned = pre[match]["rows"], "vorgerechnet", True
    source_text = EV.lueckentext(0, 0, 0, "vorgerechnet") + f" Szenario: {match}."
elif st.session_state.get("quick_for") == quick_key:
    with st.spinner("Schnellabstimmung (aus dem Zwischenspeicher)…"):
        rows, secs = _quick(items, C.P_TRAIN_QUICK, paths, seed)
    source, tuned = "schnell", True
    source_text = EV.lueckentext(secs, C.P_TRAIN_QUICK, paths, "schnell")
elif pre:
    grund = pre[next(iter(pre))]["rows"]
    with st.spinner("Die Regeln des Grundfalls werden bei Ihren Einstellungen ausgewertet …"):
        rows = _fallback(items, tuple((r["rule"], r["spec"]) for r in grund), paths, seed)
    source = "fallback"
    source_text = ("Diese Einstellung ist nicht vorgerechnet: gezeigt sind die Regeln des Grundfalls, **nicht neu abgestimmt**, ausgewertet bei Ihren Einstellungen "
                   f"({paths} Pfade, Zufalls-Seed {seed}). Der Knopf „Regeln für diese Einstellung abstimmen“ stimmt alle Regeln neu ab.")
by_name = {r["rule"]: r for r in rows}

tab_hand, tab_season, tab_years, tab_view, tab_limits = st.tabs(["✋ Von Hand", "🌱 Eine Saison", "📆 Viele Jahre", "🔭 Kurzsicht und Voraussicht", "⚠️ Grenzen"])

# --- Von Hand ----------------------------------------------------------------------------------------------------------------------------------
with tab_hand:
    st.markdown(
        "Vier Genotypen vermehren sich klonal: **SS** anfällig, **RS** resistent gegen Wirkstoff A, **SR** resistent gegen B, **RR** gegen beide. Alle Zahlen dieser Seite sind **exakte Brüche** "
        "(Abtötungsrate 9/10, Resistenzgrad 9/10); nur der Wachstumsschritt enthält eine Exponentialfunktion und steht als Zahl mit sechs Stellen."
    )
    st.markdown("**1. Überlebensanteile einer Spritzung**")
    cross_label = st.selectbox("Kreuzresistenz", ["0", "1/2", "1"], key="hand_cross")
    surv = H.survival_exact(cross=H.F(cross_label))
    st.dataframe([{"Wirkstoff": a, **{g: f"{surv[a][g]} = {float(surv[a][g]):.4g}" for g in H.GENOS}} for a in ("A", "B")], hide_index=True, width="stretch")
    st.caption("Ein Anfälliger überlebt zu 1/10. Ein Resistenter überlebt zu 1 − 9/10 · (1 − 9/10) = 91/100. Ein nur gegen den anderen Wirkstoff Resistenter überlebt zu 1/10 + Kreuzresistenz · (91/100 − 1/10).")
    st.markdown("**2. Auslese: der Anteil Resistenter wächst mit jeder Spritzung**")
    p0 = H.share_start()
    s_sus, s_res = surv["A"]["SS"], surv["A"]["RS"]
    st.dataframe(
        [{"Spritzungen k": k, "Anteil Resistenter (Bruch)": str(H.share_after(p0, k, s_sus, s_res)), "Anteil Resistenter": f"{100 * float(H.share_after(p0, k, s_sus, s_res)):.2f} %"} for k in range(6)],
        hide_index=True, width="stretch",
    )
    st.caption(f"Start: {H.S_START} anfällig und {H.R_START} resistent, also Anteil {p0}. Ohne Wachstum gilt nach k Spritzungen p·sR^k / (p·sR^k + (1 − p)·sS^k) mit sS = 1/10 und sR = 91/100.")
    st.markdown("**3. Eine Saison mit vier Wochen, ohne Wachstum**")
    sp_weeks = st.multiselect("Spritzwochen", [1, 2, 3, 4], default=[2], key="hand_weeks")
    hrows, hnet = H.exact_season(set(sp_weeks))
    st.dataframe(
        [{"Woche": r["week"], "gespritzt": "ja" if r["sprayed"] else "nein", "Anfällige": str(r["S"]), "Resistente": str(r["R"]), "Anteil Resistenter": f"{100 * float(r['share']):.2f} %",
          "Schaden (Bruch)": str(r["loss"]), "Schaden": round(float(r["loss"]), 4)} for r in hrows],
        hide_index=True, width="stretch",
    )
    st.metric("Ertrag der Saison", f"{float(hnet):.4f}", delta=f"= {hnet}", delta_color="off")
    st.caption("Wochenschaden = 100/4 · N / (N + 1/4) mit N = Anfällige + Resistente; Ertrag = 100 − Summe der Wochenschäden − 4 je Spritzung. Das Wachstum ist hier ausgeschaltet, damit alles ein Bruch bleibt.")
    st.markdown("**4. Ein Wachstumsschritt nach Ricker (Zahlen, sechs Stellen)**")
    sprayed = st.radio("In dieser Woche spritzen?", ["nein", "ja"], horizontal=True, key="hand_spray") == "ja"
    wk = H.ricker_week(0.01, 1e-4, sprayed, weeks=H.WEEKS)
    st.dataframe(
        [{"Größe": k, "Wert": f"{v:.6g}"} for k, v in (("Anfällige nach der Spritzung S₁", wk["S1"]), ("Resistente nach der Spritzung R₁", wk["R1"]), ("Gesamtdichte N₁", wk["N1"]),
                                                          ("1 − N₁/K", wk["g"]), ("Wachstumsfaktor Anfällige exp(0,45·(1 − N₁))", wk["fS"]),
                                                          ("Wachstumsfaktor Resistente exp(0,45·0,7·(1 − N₁))", wk["fR"]), ("S₂", wk["S2"]), ("R₂", wk["R2"]),
                                                          ("Mittel der Gesamtdichte (N₁ + N₂)/2", wk["Nm"]), ("Wochenschaden (Ertragspunkte, bei 4 Wochen je Saison)", wk["loss"]))],
        hide_index=True, width="stretch",
    )
    st.caption("Start S = 1/100, R = 1/10000, Fitnesskosten 0,3 (Wachstumsfaktor der Resistenten 0,7), Wachstumsrate 0,45, Wetterfaktor 1.")

# --- Eine Saison -------------------------------------------------------------------------------------------------------------------------------
with tab_season:
    st.markdown("## 🌱 Eine Saison, Woche für Woche")
    names = [r["rule"] for r in rows]
    c1, c2 = st.columns([2, 1])
    pick = c1.selectbox("Regel", names, index=min(1, len(names) - 1), key="season_rule")
    year = c2.slider("Jahr", 1, C.YEARS, 1, key="season_year")
    sel = by_name[pick]
    ss = _season(items, pick, sel["spec"], int(year))
    st.caption("Ohne Zufall: Wetterfaktor 1, Zuwanderung 1, exakte Messung. Gezeigt ist die behandelte Fläche; die Regel läuft mit den Werten der Tabelle „Viele Jahre“ (" + ("abgestimmt" if tuned else "Grundfall, nicht neu abgestimmt") + ").")
    m1, m2, m3 = st.columns(3)
    m1.metric("Ertrag in diesem Jahr", f"{ss['net']:.1f}")
    m2.metric("Spritzungen in diesem Jahr", f"{ss['sprays']}")
    m3.metric("Resistenzanteil gegen A am Saisonende", f"{100 * ss['resA']:.3f} %")
    st.plotly_chart(V.fig_season_density(ss["n"], ss["weeks"], f"{pick}: Jahr {year}"), width="stretch", key="season_density")
    st.plotly_chart(V.fig_season_loss(ss["loss"], "Wochenschaden"), width="stretch", key="season_loss")

# --- Viele Jahre -------------------------------------------------------------------------------------------------------------------------------
with tab_years:
    st.markdown("## 📆 Viele Jahre: die Regeln im Vergleich")
    st.markdown(source_text)
    if match is not None:
        st.caption("Diese Einstellung ist vorgerechnet: eine Schnellabstimmung ist nicht nötig. Bei anderen Reglerwerten erscheint hier der Knopf „Regeln für diese Einstellung abstimmen“.")
    elif st.button("🔧 Regeln für diese Einstellung abstimmen (Näherung, einige Sekunden bis eine Minute)", key="quick_button",
                   help="Stimmt alle Regeln auf wenigen Trainingspfaden neu ab."):
        with st.spinner("Alle Regeln werden abgestimmt (Näherung, 60 Trainingspfade) …"):
            _quick(items, C.P_TRAIN_QUICK, paths, seed)
        st.session_state["quick_for"] = quick_key
        st.rerun()
    if rows:
        st.dataframe(
            [{"Regel": r["rule"], total_label: round(r["total"], 1), "SE": round(r["se"], 2), "gegen nie": round(r["d_nie"], 1), "gegen Kalender": round(r["d_cal"], 1),
              "SE gegen Kalender": round(r["d_cal_se"], 2), "Jahre 1 bis 5": round(r["blk1"], 1), "Jahre 6 bis 10": round(r["blk2"], 1), "Jahre 11 bis 15": round(r["blk3"], 1),
              "Jahre 16 bis 20": round(r["blk4"], 1), "Spritzungen je Jahr": round(r["sprays"], 1), "Resistenz A im Jahr 20": 100 * r["resA20"], "Parameter": r["params"]} for r in rows],
            hide_index=True, width="stretch",
            column_config={"Resistenz A im Jahr 20": st.column_config.NumberColumn(format="%.3f %%")},
        )
        st.caption("Ertragspunkte (100 = Höchstertrag je Jahr), Summe über 20 Jahre; SE = Standardfehler über die Pfade, bei „gegen …“ der gepaarte Unterschied. Die Blöcke zeigen den Ertrag im Mittel der Jahre (ohne Abzinsung): das Jahr 1 beginnt mit einem neuen Wirkstoff und enthält den Anfangsvorteil.")
        chosen = st.multiselect("Regeln im Diagramm", names, default=[n for n in C.DEFAULT_PLOT_RULES if n in names], key="plot_rules")
        if chosen:
            band_name = st.selectbox("Pfadband (10 bis 90 %) für", chosen, key="band_rule")
            curves = {n: _curves(items, n, by_name[n]["spec"], paths, seed) for n in chosen}
            st.plotly_chart(V.fig_years({n: c["mean"] for n, c in curves.items()}, (curves[band_name]["lo"], curves[band_name]["hi"]), band_name), width="stretch", key="years_chart")
            st.plotly_chart(V.fig_resistance({n: c["resA"] for n, c in curves.items()}, {n: c["resB"] for n, c in curves.items() if c["mode"] != "A"}), width="stretch", key="res_chart")
            st.caption(f"Kurven: {paths} Pfade, Zufalls-Seed {seed}, Regeln mit den Werten der Tabelle (die Tabelle selbst: siehe Herkunft oben).")

# --- Kurzsicht und Voraussicht -----------------------------------------------------------------------------------------------------------------
with tab_view:
    st.markdown("## 🔭 Kurzsicht gegen Voraussicht")
    st.markdown("**Auf wie viele Jahre wird abgestimmt?** (Grundfall, vorgerechnet)")
    hz = _horizont()
    pre_first = next(iter(pre.values()), None)
    if hz and pre_first:
        nie0 = pre_first["rows"][0]["total"]
        st.plotly_chart(V.fig_horizon(hz, nie0), width="stretch", key="horizon_chart")
        st.caption("Summe über 20 Jahre bei der Abstimmung auf H Jahre (Schwelle und Kalender), Kontrollabstand 2 oder 4 Wochen abgestimmt; 250 Trainings-, 1000 Auswertungspfade. Kurze Horizonte (1 bis 10 Jahre) wählen kleine Schwellen (0,004 bis 0,021), erst die Abstimmung auf 20 Jahre eine große (0,74).")
    st.markdown("**Schwellenprofil bei Ihren Einstellungen** (live)")
    v_pick = st.selectbox("Kontrollabstand (Wochen)", [1, 2, 4], index=1, key="profile_v")
    with st.spinner(f"Schwellenprofil wird gerechnet ({C.LIVE_PROFILE_PATHS} Pfade) …"):
        S_prof, tot_prof, nie_prof = _profile(items, int(v_pick), min(paths, C.LIVE_PROFILE_PATHS), seed)
    st.plotly_chart(V.fig_profile(S_prof, tot_prof, nie_prof), width="stretch", key="profile_chart")
    st.caption(f"{total_label} in Abhängigkeit von der Schwelle, {min(paths, C.LIVE_PROFILE_PATHS)} Pfade, Zufalls-Seed {seed}; rechts „fast nie spritzen“.")

    st.markdown("**Referenz: volle Voraussicht (Rückwärtsrechnung)**")
    dpw = None
    for name, sc in pre.items():
        s0 = sc["settings"]
        if all(abs(s0[k] - float(settings[k])) < 1e-9 for k in ("phi", "cost", "kill", "m", "spray")) and name in dp_werte:
            dpw = dp_werte[name]
            break
    if dpw is not None:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Rückwärtsrechnung", f"{dpw['dp']:.0f}")
        d2.metric("Jahresweise beste Antwort", f"{dpw['myopic']:.0f}")
        d3.metric("Bester Kalender", f"{dpw['cal']:.0f}")
        d4.metric("Beste Schwelle", f"{dpw['thr']:.0f}")
        st.caption("Deterministische Instanz (kein Rauschen, Kontrollen gratis, kein Refugium, keine Abzinsung), ein Wirkstoff; nie spritzen: " + f"{dpw['nie']:.0f}. "
                   "Die jahresweise beste Antwort kennt den Zustand, plant aber nur das laufende Jahr.")
    else:
        st.info("Für diese Werte von φ, c, Abtötungsrate, Zuwanderung und Spritzkosten ist die Rückwärtsrechnung nicht vorgerechnet (Presets und Grundfall sind es).")
    if dp_politik:
        st.plotly_chart(V.fig_dp(dp_politik), width="stretch", key="dp_chart")
        st.caption("Grundfall, ein Wirkstoff: die Rückwärtsrechnung spritzt in Stößen und lässt Jahre aus, in denen die Fitnesskosten die Resistenz abbauen; die jahresweise beste Antwort spritzt jedes Jahr.")
    if dp2_politik:
        st.plotly_chart(V.fig_dp2(dp2_politik), width="stretch", key="dp2_chart")
        st.caption("Grundfall, zwei Wirkstoffe (ohne Doppelresistente): die Politik der Rückwärtsrechnung.")

    st.markdown("**Wo kippt es? Raster Resistenzgrad × Fitnesskosten** (vorgerechnet)")
    grid = _raster()
    if grid:
        metrics = {
            "Schwelle S, auf 20 Jahre abgestimmt": lambda r: (np.log10(r["S20"]), f"{r['S20']:.3g}"),
            "Schwelle S, auf 1 Jahr abgestimmt": lambda r: (np.log10(r["S1"]), f"{r['S1']:.3g}"),
            "Gewinn der Voraussicht (Summe S 20 Jahre minus S 1 Jahr)": lambda r: (r["sum20"] - r["sum1"], f"{r['sum20'] - r['sum1']:.0f}"),
            "Kurzsicht schlechter als nie spritzen um (Summe nie minus Summe S 1 Jahr)": lambda r: (r["nie"] - r["sum1"], f"{r['nie'] - r['sum1']:.0f}"),
        }
        metric_name = st.selectbox("Kennzahl", list(metrics), key="raster_metric")
        near_phi = min(C.GRID_PHI, key=lambda x: abs(x - float(settings["phi"])))
        near_cost = min(C.GRID_COST, key=lambda x: abs(x - float(settings["cost"])))
        st.plotly_chart(V.fig_raster(grid, metrics[metric_name], current=(near_phi, near_cost), title=metric_name), width="stretch", key="raster_chart")
        cell = G.raster_lookup(grid, near_phi, near_cost)
        kind, text = EV.kipp_verdikt(near_phi, near_cost, cell["S1"], cell["S20"], cell["sum1"], cell["sum20"])
        getattr(st, kind)(text)
        others = {k: v for k, v in settings.items() if k not in ("phi", "cost", "seed", "paths", "q", "horizon")}
        defaults = {k: v for k, v in DEFAULTS.items() if k in others}
        if others != defaults:
            st.warning("Ihre übrigen Regler weichen von den Annahmen des Rasters ab (Standardwerte, Kontrollabstand 2 Wochen): das Raster gilt für die Standardwerte, ◎ markiert nur die nächstliegende Zelle von φ und c.")
        st.caption("Vorgerechnet mit 250 Trainings- und 1000 Auswertungspfaden, Kontrollabstand 2 Wochen. Die Schwellen S sind Dichten (Kapazität = 1); 1,5 heißt: die Abstimmung wählt den obersten Gitterwert, praktisch „nie spritzen“ (Messrauschen löst noch selten aus, und Kontrollen kosten weiter). Farbe: Rot = große Zahl (Gedächtnis-Dilemma), Grün = kleine Zahl.")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------
with tab_limits:
    st.markdown(TX.ASSUMPTIONS_MARKDOWN)

# --- Kernfragen --------------------------------------------------------------------------------------------------------------------------------
st.markdown("## ❓ Kernfragen")
verdict_texts = []
if tuned:
    st.markdown("**Was kostet Kurzsicht?**")
    kind, text = EV.kurzsicht_verdikt(by_name[C.R_THR1]["total"], by_name[C.R_NEVER]["total"], by_name[C.R_THRH]["total"])
    getattr(st, kind)(text)
    verdict_texts.append(text)
    st.markdown("**Lohnen Pausen?**")
    pr = by_name[C.R_PAUSE]
    kind, text = EV.pausen_verdikt(pr["d_cal"], pr["d_cal_se"], by_name[C.R_CAL]["total"])
    getattr(st, kind)(text)
    verdict_texts.append(text)
    st.markdown("**Hilft der Wirkstoffwechsel?**")
    rot = max(by_name[n]["total"] for n in (C.R_CAL_YEAR, C.R_CAL_SPRAY, C.R_THR_YEAR, C.R_THR_SPRAY))
    single = max(by_name[n]["total"] for n in (C.R_CAL, C.R_THR1, C.R_THR5, C.R_THRH, C.R_CAP, C.R_PAUSE))
    kind, text = EV.wechsel_verdikt(rot, single, float(settings["cross"]))
    getattr(st, kind)(text)
    verdict_texts.append(text)
else:
    st.info("Für diese Einstellung ist die Regeltabelle nicht abgestimmt: die drei Fragen zu Kurzsicht, Pausen und Wirkstoffwechsel brauchen abgestimmte Regeln. Im Tab „Viele Jahre“ stimmt der Knopf alle Regeln neu ab.")
st.markdown("**Was bringt volle Voraussicht?**")
if dpw is not None:
    kind, text = EV.voraussicht_verdikt(dpw["dp"], dpw["myopic"], dpw["cal"])
    if disc < 1.0:
        text += " Die Referenz rechnet ohne Abzinsung."
    getattr(st, kind)(text)
    verdict_texts.append(text)
else:
    st.info("Die Rückwärtsrechnung ist für diese Werte nicht vorgerechnet.")

pdf_bytes = build_pdf(
    {"Resistenzgrad": settings["phi"], "Fitnesskosten": settings["cost"], "Abtötungsrate": settings["kill"], "Zuwanderung": settings["m"], "Spritzkosten": settings["spray"],
     "Kontrollkosten": settings["scout"], "Messrauschen": settings["meas"], "Kreuzresistenz": settings["cross"], "Doppelresistenz (Zehnerpotenz)": settings["rr"],
     "Refugium": settings["q"], "Horizont": settings["horizon"], "Zinsfaktor": settings["disc"], "Zufalls-Seed": seed, "Pfade": paths},
    rows, source_text.replace("**", ""), verdict_texts,
)
st.download_button("📄 Ergebnis als PDF", data=pdf_bytes, file_name="schaedlingsbekaempfung-resistenz.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")

# --- Expander ----------------------------------------------------------------------------------------------------------------------------------
with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich"):
    st.markdown(TX.METHODS_MARKDOWN)

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/schaedling-resistenz-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind vier Genotypen mit Dichten $n_g$ ($g\in\{SS,RS,SR,RR\}$), zwei Wirkstoffe und eine Regel, die Woche für Woche über das Spritzen entscheidet. Gesucht ist die Regel mit dem größten (abgezinsten) Gesamtertrag über $Y$ Jahre, wenn die Resistenz von Jahr zu Jahr zunimmt.

| Symbol | Bedeutung |
|---|---|
| $n_g$, $N=\sum_g n_g$ | Dichte des Genotyps $g$ und Gesamtdichte (Kapazität $K=1$) |
| $r$, $\rho_y$, $c$ | Wachstumsrate je Woche, Wetterfaktor des Jahres, Fitnesskosten je Resistenz |
| $f_g=(1-c)^{\lvert g\rvert}$ | Wachstumsfaktor, $\lvert g\rvert$ = Zahl der Resistenzen des Genotyps |
| $\kappa$, $\varphi$, $\chi$ | Abtötungsrate, Resistenzgrad, Kreuzresistenz |
| $s_{a,g}$ | Überlebensanteil des Genotyps $g$ bei Wirkstoff $a\in\{A,B\}$ |
| $b$, $h$, $W$ | größter Verlust, Halbsättigung des Schadens, Wochen je Saison |
| $c_s$, $c_k$, $q$ | Kosten einer Spritzung, einer Kontrolle, Refugium-Anteil |
| $\nu$, $m$, $\mu_y$, $\pi_g$ | Überwinterung, Zuwanderung, Streufaktor der Zuwanderung, Genotypanteile der Zuwanderer |
| $\delta$, $H$ | Zinsfaktor je Jahr, Planungshorizont der Abstimmung |
| $k$, $w_0$, $S$, $v$ | Kalender: Abstand und Start; Schwelle und Kontrollabstand |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/schaedling-resistenz-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(r"""$$
n_g \leftarrow n_g\; s_{a,g}\quad\text{(Spritzung mit Wirkstoff } a\text{, vor dem Wachstum)}
$$""")
    st.markdown(r"""$$
s_{a,g}=\begin{cases}1-\kappa & g \text{ gegen keinen der beiden Wirkstoffe resistent}\\ 1-\kappa(1-\varphi) & g \text{ gegen } a \text{ resistent}\\ (1-\kappa)+\chi\,\big[(1-\kappa(1-\varphi))-(1-\kappa)\big] & g \text{ nur gegen den anderen Wirkstoff resistent}\end{cases}
$$""")
    st.markdown(r"""$$
n_g^{+}=n_g\,\exp\!\Big(r\,\rho_y\,f_g\,\big(1-N/K\big)\Big),\qquad f_g=(1-c)^{\lvert g\rvert}
$$""")
    st.markdown(r"""$$
L_{y,t}=\frac{b}{W}\,\frac{\bar N}{\bar N+h},\qquad \bar N=\tfrac12\big(N^{\text{nach Spritzung}}+N^{+}\big)
$$""")
    st.markdown(r"""$$
E_y=100\Big(1-\sum_{t=0}^{W-1}L_{y,t}\Big)-c_s\,(1-q)\,n^{\text{Spritz}}_y-c_k\,n^{\text{Kontr}}_y
$$""")
    st.markdown(r"""$$
n_{g,y+1,0}=\nu\,\big[(1-q)\,n^{\text{behandelt}}_{g,y,W}+q\,n^{\text{Refugium}}_{g,y,W}\big]+m\,\mu_y\,\pi_g
$$""")
    st.markdown(r"""$$
\max_{\text{Regel}}\ \sum_{y=1}^{Y}\delta^{\,y-1}\,\mathbb{E}\,[E_y]
$$""")
    st.markdown(r"""$$
\text{Kalender: spritze in Woche } t \iff t\ge w_0\ \text{und}\ (t-w_0)\bmod k=0
$$""")
    st.markdown(r"""$$
\text{Schwelle: spritze in Woche } t \iff t \bmod v=0\ \text{und}\ \hat N_t=N_t\,e^{\sigma_k z_t}>S,\quad z_t\sim\mathcal N(0,1)
$$""")
    st.markdown(r"""$$
p_k=\frac{p\,s_R^{\,k}}{p\,s_R^{\,k}+(1-p)\,s_S^{\,k}}\quad\text{(Anteil Resistenter nach } k \text{ Spritzungen ohne Wachstum)}
$$""")
    st.markdown(r"""$$
V_t(x)=\max_{a\in\{0,1\}}\Big\{-C_a(x)+V_{t+1}\big(T_a(x)\big)\Big\},\qquad x=(\log_{10}S,\ \log_{10}R)
$$""")
    st.markdown(TX.MATH_NOTE)
    st.markdown(
        "In der Rückwärtsrechnung ist $T_a$ die Wochengleichung ohne Zufall (erst Spritzung, dann Wachstum), $C_a$ der Wochenschaden plus $c_s\\,a$; in der letzten Woche eines Jahres ist $V_{t+1}$ der Wert nach dem Winter. "
        "Die Wertfunktion liegt auf einem Gitter von $0{,}05$ Dekaden und wird bilinear interpoliert."
    )

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(TX.ASSUMPTIONS_MARKDOWN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
