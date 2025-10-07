# uncertainty_profile.py
# Streamlit app: Personal Uncertainty Profile (Swedish Election Futures 2026)
# - Sliders for Q1–Q10 with tooltips (motiveringar)
# - Radar chart (matplotlib), bar chart, priors overlay
# - Export PNG/SVG/CSV
# - Shareable URL via st.query_params (?p=comma-separated 10 ints)

import io
import math
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Uncertainty Profile — Sweden 2026", layout="wide")

# ---------- Model questions & priors ----------
PRIORS: List[int] = [62, 18, 45, 35, 48, 12, 30, 40, 42, 28]  # % priors (v2.0)

QUESTIONS = [
    ("Q1", "Regeringsbildning > 50 dagar",
     "Mäter systemets förmåga att producera handlingskraft (fragmentering ↔ styrbarhet)."),
    ("Q2", "S+M tolereras i regering inom 1 år",
     "Indikator på blockpolitikens kollaps och kriskoalitioners plausibilitet."),
    ("Q3", "SD innehar ministerpost före 2027-01-01",
     "Normalisering i regeringsmakten; omdefinierar högerblockets arkitektur."),
    ("Q4", "C pekar ut annan statsministerkandidat före valdagen",
     "Mittenpolitikens hävstång över regeringsbildningen."),
    ("Q5", "Valdeltagande > 85,0 %",
     "Demokratins vitalitet (mobilisering vs apati)."),
    ("Q6", "Nytt parti ≥ 4,0 %",
     "Partisystemets innovations- och sprickbenägenhet."),
    ("Q7", "Första budgeten antas blocköverskridande",
     "Pragmatism under tryck; styrbarhet > renlärighet."),
    ("Q8", "Regeringsparti byter partiledare 2026–27",
     "Politiskt slitage/krishantering i regeringsposition."),
    ("Q9", "Störst i EU-val 2029 = störst i riksdagsval 2026",
     "Partilojalitet över valcykler och volatilitet i väljarkåren."),
    ("Q10", "Minst ett nuvarande riksdagsparti < 4,0 % 2026",
     "Systemets dynamik och utslagningstryck på småpartier."),
]
LABELS = [q[0] for q in QUESTIONS]

# ---------- Query param helpers (new API only) ----------
def read_query_profile() -> Optional[List[int]]:
    """Read ?p=62,18,...,28 from st.query_params (new stable API)."""
    qp = st.query_params.get("p", None)
    if not qp:
        return None
    try:
        vals = [int(float(x)) for x in qp.split(",")]
        if len(vals) == 10 and all(0 <= v <= 100 for v in vals):
            return vals
    except Exception:
        pass
    return None

def write_query_profile(values: List[int]) -> None:
    """Set/update ?p=... using st.query_params."""
    st.query_params["p"] = ",".join(str(int(v)) for v in values)

# ---------- Radar plot ----------
def radar_png_svg(values: List[int],
                  priors: Optional[List[int]] = None,
                  title: str = "Uncertainty Profile (Q1–Q10)"):
    assert len(values) == 10
    angles = np.linspace(0, 2 * math.pi, 10, endpoint=False).tolist()
    vals = list(values)
    vals_loop = vals + vals[:1]
    ang_loop = angles + angles[:1]

    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), LABELS)

    # Sober grid aesthetics (no explicit colors)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.7)
    ax.xaxis.grid(True, linestyle='-', linewidth=0.7)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax.set_title(title, va="bottom", fontsize=14)

    # Optional priors overlay
    if priors is not None:
        pv = list(priors)
        pv_loop = pv + pv[:1]
        ax.plot(ang_loop, pv_loop, linewidth=1.5, linestyle='--')
        ax.fill(ang_loop, pv_loop, alpha=0.08)

    # Profile line + fill + labels
    ax.plot(ang_loop, vals_loop, linewidth=2.6, marker='o', markersize=5)
    ax.fill(ang_loop, vals_loop, alpha=0.14)
    for a, v in zip(angles, values):
        ax.text(a, v + 3, f"{int(round(v))}%", ha="center", va="center", fontsize=9)

    plt.tight_layout()

    # Export buffers
    buf_png, buf_svg = io.BytesIO(), io.BytesIO()
    fig.savefig(buf_png, bbox_inches="tight", dpi=220, format="png")
    fig.savefig(buf_svg, bbox_inches="tight", format="svg")
    buf_png.seek(0); buf_svg.seek(0)
    return fig, buf_png, buf_svg

# ---------- UI ----------
st.title("Swedish Election Futures 2026 — Personal Uncertainty Profile")
st.caption("Sätt dina sannolikheter (0–100 %) för Q1–Q10 och få en sober radar-visualisering.")

with st.sidebar:
    st.header("Inställningar")
    imported = read_query_profile()
    use_seed = st.toggle("Fyll med priors (v2.0)", value=(imported is None))
    overlay_priors = st.toggle("Visa priors som overlay", value=True)
    st.markdown("---")
    st.write("**Snabblägen**")
    if st.button("Slumpa profil (för demo)"):
        rnd = np.random.default_rng()
        rand_vals = [int(x) for x in rnd.integers(0, 101, size=10)]
        write_query_profile(rand_vals)
        st.rerun()
    if st.button("Återställ URL-parametrar"):
        st.query_params.clear()
        st.rerun()

col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.subheader("Din bedömning")
    values: List[int] = []
    src = imported if imported is not None else (PRIORS if use_seed else [50]*10)
    for i, (qid, title, mot) in enumerate(QUESTIONS):
        val = st.slider(f"{qid} — {title}",
                        min_value=0, max_value=100,
                        value=int(src[i]), step=1,
                        help=mot, key=f"q_{i}")
        values.append(val)

    # Shareable link
    write_query_profile(values)
    st.info("Permalänk uppdateras i adressfältet (`?p=`) för att dela din profil.")

with col_right:
    st.subheader("Radar")
    fig, buf_png, buf_svg = radar_png_svg(values,
                                          priors=(PRIORS if overlay_priors else None),
                                          title="Uncertainty Profile (Q1–Q10)")
    st.pyplot(fig, clear_figure=True)
    st.download_button("Ladda ned PNG", data=buf_png,
                       file_name="uncertainty_profile_radar.png", mime="image/png")
    st.download_button("Ladda ned SVG", data=buf_svg,
                       file_name="uncertainty_profile_radar.svg", mime="image/svg+xml")

st.markdown("---")
st.subheader("Tabell & Bar")

df = pd.DataFrame({
    "Fråga": LABELS,
    "Beskrivning": [q[1] for q in QUESTIONS],
    "Motivering": [q[2] for q in QUESTIONS],
    "Priors (%)": PRIORS,
    "Din profil (%)": values,
    "Delta (profil − priors, %-enheter)": [round(v - p, 1) for v, p in zip(values, PRIORS)],
})
st.dataframe(df, use_container_width=True)

st.bar_chart(pd.DataFrame({"Fråga": LABELS, "Sannolikhet (%)": values}),
             x="Fråga", y="Sannolikhet (%)")

# CSV export
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Ladda ned CSV", data=csv_bytes,
                   file_name="uncertainty_profile.csv", mime="text/csv")

st.markdown("—")
st.caption("Tips: Lägg till veckosnapshots och jämför radar över tid för att se hur nyheter flyttar dina bedömningar.")
