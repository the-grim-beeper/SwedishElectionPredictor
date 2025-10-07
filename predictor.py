# app.py
import json
import math
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

DB_PATH = "futures_votes.sqlite"

QUESTIONS = [
    "Q1: Regeringsbildning > 50 dagar (proxy för längd)",
    "Q2: S+M tolereras i regering inom 1 år",
    "Q3: SD innehar minst en ministerpost före 2027-01-01",
    "Q4: C pekar ut annan statsministerkandidat före valdagen",
    "Q5: Valdeltagande > 85,0 %",
    "Q6: Nytt parti når ≥ 4,0 %",
    "Q7: Första budgeten antas blocköverskridande",
    "Q8: Regeringsparti byter partiledare (2026–2027)",
    "Q9: Största parti i EU-val 2029 = största i riksdagsval 2026",
    "Q10: Minst ett nuvarande riksdagsparti < 4,0 % 2026",
]

DEFAULT_PRIORS = [62, 18, 45, 35, 48, 12, 30, 40, 42, 28]  # percent, v2.0

# ---------- storage ----------
def ensure_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_label TEXT,
                created_at TEXT,
                q1 REAL, q2 REAL, q3 REAL, q4 REAL, q5 REAL,
                q6 REAL, q7 REAL, q8 REAL, q9 REAL, q10 REAL
            );
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS priors (
                id INTEGER PRIMARY KEY CHECK (id=1),
                payload TEXT
            );
            """
        )
        # seed priors if empty
        c.execute("SELECT COUNT(*) FROM priors;")
        if c.fetchone()[0] == 0:
            payload = json.dumps({"priors": DEFAULT_PRIORS})
            c.execute("INSERT INTO priors (id, payload) VALUES (1, ?);", (payload,))
        conn.commit()


def load_priors():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT payload FROM priors WHERE id=1;")
        row = c.fetchone()
        priors = json.loads(row[0])["priors"]
    return priors


def save_priors(new_priors):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE priors SET payload=? WHERE id=1;",
            (json.dumps({"priors": new_priors}),),
        )
        conn.commit()


def insert_vote(user_label, probs):
    # probs in 0..100
    created_at = datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO votes (
                user_label, created_at,
                q1,q2,q3,q4,q5,q6,q7,q8,q9,q10
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (user_label, created_at, *probs),
        )
        conn.commit()


def load_votes_df():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM votes ORDER BY created_at DESC;", conn)
    return df


# ---------- visuals ----------
def radar_chart(labels, values_percent, title="Crowd radar"):
    # values_percent: list of 0..100
    angles = np.linspace(0, 2 * math.pi, len(labels), endpoint=False).tolist()
    values_loop = values_percent + values_percent[:1]
    angles_loop = angles + angles[:1]

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), labels)
    ax.plot(angles_loop, values_loop, linewidth=2)
    ax.fill(angles_loop, values_loop, alpha=0.15)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])
    ax.set_title(title, va="bottom")
    for angle, val in zip(angles, values_percent):
        ax.text(angle, val + 3, f"{int(round(val))}%", ha="center", va="center", fontsize=8)
    st.pyplot(fig, clear_figure=True)


# ---------- app ----------
def main():
    st.set_page_config(page_title="Swedish Election Futures 2026", layout="wide")
    ensure_db()

    st.title("Swedish Election Futures 2026 — Crowd Priors")
    st.caption("Uppdatera dina sannolikheter (i procent). Varje inskickning sparas och snittas över alla användare.")

    priors = load_priors()
    with st.sidebar:
        st.header("Priors (admin)")
        st.caption("Förifyllda värden i formuläret; påverkar inte redan inlämnade röster.")
        pri_edit = st.toggle("Redigera priors")
        editable = pri_edit
        new_priors = []
        for i, q in enumerate(QUESTIONS):
            val = priors[i]
            if editable:
                val = st.slider(q, 0, 100, int(val), step=1, key=f"prior_{i}")
            else:
                st.write(f"{q}: **{val}%**")
            new_priors.append(int(val))
        if editable and st.button("Spara priors"):
            save_priors(new_priors)
            st.success("Priors uppdaterade.")

        st.divider()
        if st.checkbox("Exportera alla röster som CSV"):
            df_all = load_votes_df()
            csv = df_all.to_csv(index=False).encode("utf-8")
            st.download_button("Ladda ned CSV", data=csv, file_name="futures_votes.csv", mime="text/csv")

        if st.checkbox("⚠️ Admin: Töm databasen (röster)"):
            if st.button("Bekräfta återställning"):
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute("DELETE FROM votes;")
                st.warning("Alla röster raderade.")

    st.subheader("Din inlämning")
    col_user, col_prefill = st.columns([1, 1])
    with col_user:
        user_label = st.text_input("Valfri signatur (för logg/etiketter)", placeholder="anonym")
    with col_prefill:
        use_priors = st.toggle("Fyll formulär med priors", value=True)

    sliders = []
    st.write("Sätt sannolikheter (0–100 %):")
    for i, q in enumerate(QUESTIONS):
        default = int(priors[i]) if use_priors else 50
        sliders.append(st.slider(q, 0, 100, default, step=1, key=f"q_{i}"))

    if st.button("Skicka in mina sannolikheter"):
        insert_vote(user_label.strip() or "anonym", sliders)
        st.success("Inlämnat! Tack.")

    # --- aggregates ---
    df = load_votes_df()
    if df.empty:
        st.info("Inga inlämningar ännu. Använd formuläret ovan för att skapa första datapunkten.")
        return

    # compute crowd mean and show charts
    probs_cols = [f"q{i}" for i in range(1, 11)]
    crowd_mean = df[probs_cols].mean().astype(float).tolist()

    st.subheader("Crowd-genomsnitt")
    agg = pd.DataFrame({
        "Fråga": QUESTIONS,
        "Genomsnitt (%)": [round(x, 1) for x in crowd_mean]
    })
    st.dataframe(agg, use_container_width=True)

    # bar
    st.write("### Bar diagram (crowd)")
    bar_df = pd.DataFrame({"Fråga": QUESTIONS, "Sannolikhet (%)": crowd_mean})
    st.bar_chart(bar_df, x="Fråga", y="Sannolikhet (%)")

    # radar
    st.write("### Radar (crowd)")
    radar_chart(QUESTIONS, [float(x) for x in crowd_mean], title="Crowd radar — Swedish Election Futures 2026")

    # compare to priors
    st.write("### Jämförelse: priors vs crowd")
    comp = pd.DataFrame({
        "Fråga": QUESTIONS,
        "Priors (%)": priors,
        "Crowd (%)": [round(x, 1) for x in crowd_mean],
        "Delta (Crowd − Priors, %-enheter)": [round(c - p, 1) for c, p in zip(crowd_mean, priors)],
    })
    st.dataframe(comp, use_container_width=True)

    # show latest submissions
    st.write("### Senaste inlämningar")
    nice = df[["created_at", "user_label"] + probs_cols].copy()
    nice.rename(columns={
        "created_at": "UTC-tid",
        "user_label": "Signatur",
        **{f"q{i}": QUESTIONS[i-1] for i in range(1, 11)}
    }, inplace=True)
    st.dataframe(nice.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
