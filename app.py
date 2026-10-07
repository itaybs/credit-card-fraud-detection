"""Credit Card Fraud Detection dashboard (Hebrew, RTL).

Launch with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import BINARY_FEATURES, dataset_summary, load_data
from model import (
    ARTIFACT_PATH,
    CV_FOLDS,
    evaluate,
    explain,
    load_artifacts,
    predict_proba,
    save_artifacts,
    train_and_evaluate,
)

st.set_page_config(page_title="זיהוי הונאות בכרטיסי אשראי", page_icon="💳", layout="wide")

# ---------------------------------------------------------------- constants
MERCHANT_HE = {
    "Clothing": "ביגוד",
    "Electronics": "אלקטרוניקה",
    "Food": "מזון ומסעדות",
    "Grocery": "סופרמרקט",
    "Travel": "תיירות ונסיעות",
}
FEATURE_HE = {
    "amount": "סכום העסקה",
    "transaction_hour": "שעת העסקה",
    "merchant_category": "קטגוריית בית העסק",
    "foreign_transaction": 'עסקה בחו"ל',
    "location_mismatch": "אי-התאמת מיקום",
    "device_trust_score": "ציון אמינות המכשיר",
    "velocity_last_24h": "עסקאות ב-24 השעות האחרונות",
    "cardholder_age": "גיל בעל הכרטיס",
}
INPUT_DEFAULTS = {
    "amount": 120.0,
    "transaction_hour": 12,
    "merchant_category": "Grocery",
    "foreign_transaction": False,
    "location_mismatch": False,
    "device_trust_score": 65,
    "velocity_last_24h": 2,
    "cardholder_age": 44,
}

# Palette
NAVY, BLUE, GRAY = "#1E2A4A", "#3B5BDB", "#98A2B3"
RED, RED_BG, RED_TXT = "#E5484D", "#FDECEC", "#B42318"
GREEN, GREEN_BG, GREEN_TXT = "#2F9E5B", "#E8F6EE", "#1E6B3E"
AMBER, AMBER_BG, AMBER_TXT = "#F5A524", "#FFF6E5", "#8A5A00"


def feature_label(name: str) -> str:
    if name.startswith("merchant_category_"):
        return f"בית עסק: {MERCHANT_HE.get(name.split('_')[-1], name)}"
    return FEATURE_HE.get(name, name)


# ---------------------------------------------------------------- styling
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;600;700;800&display=swap');

.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp textarea,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp td, .stApp th {{
  font-family: 'Heebo', sans-serif;
}}

/* RTL layout for content + sidebar; charts and slider tracks stay LTR */
[data-testid="stMain"], [data-testid="stSidebar"] {{ direction: rtl; text-align: right; }}
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] {{
  text-align: right;
}}
[data-testid="stPlotlyChart"], .js-plotly-plot {{ direction: ltr; }}
[data-testid="stSlider"] [role="group"], [data-testid="stSlider"] [data-baseweb="slider"] {{ direction: ltr; }}
[data-testid="stNumberInput"] input {{ direction: ltr; text-align: right; }}
[data-baseweb="popover"] li {{ direction: rtl; text-align: right; }}
[data-testid="stTooltipContent"] {{ direction: rtl; text-align: right; }}

.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}

/* Bordered containers -> cards */
[data-testid="stVerticalBlockBorderWrapper"] {{
  border-radius: 16px !important; border-color: #E6E9F0 !important; background: #FFFFFF;
  box-shadow: 0 1px 3px rgba(16, 24, 40, .05);
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid #E6E9F0; }}
.stTabs [data-baseweb="tab"] {{ padding: 10px 18px; font-weight: 600; font-size: 1rem; }}

/* Hero */
.hero {{
  background: linear-gradient(120deg, {NAVY} 0%, #2B3F7A 60%, {BLUE} 100%);
  color: #fff; border-radius: 18px; padding: 26px 30px; margin-bottom: 18px;
  box-shadow: 0 8px 24px rgba(30, 42, 74, .18);
}}
.hero h1 {{ color: #fff; font-size: 1.9rem; font-weight: 800; margin: 0 0 6px 0; padding: 0; }}
.hero p {{ color: #D6DEF5; margin: 0; font-size: 1rem; }}
.hero .chips {{ margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }}
.hero .chip {{
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22);
  padding: 4px 12px; border-radius: 999px; font-size: .85rem; color: #fff;
}}

/* Section titles */
.section-title {{
  font-size: 1.15rem; font-weight: 700; color: {NAVY}; margin: 22px 0 10px 0;
  padding-right: 10px; border-right: 4px solid {BLUE}; line-height: 1.3;
}}
.section-sub {{ color: #667085; font-size: .9rem; margin: -6px 0 12px 0; }}

/* KPI cards */
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; }}
.kpi {{
  background: #fff; border: 1px solid #E6E9F0; border-radius: 16px; padding: 16px 18px;
  border-top: 4px solid var(--accent); box-shadow: 0 1px 3px rgba(16,24,40,.05);
}}
.kpi-label {{ font-size: .9rem; font-weight: 600; color: #475467; }}
.kpi-value {{ font-size: 2rem; font-weight: 800; color: {NAVY}; line-height: 1.25; margin-top: 4px; }}
.kpi-sub {{ font-size: .8rem; color: #667085; margin-top: 2px; }}
.kpi-tint {{ border-top: 0; background: var(--bg); border-color: transparent; }}
.kpi-tint .kpi-value {{ color: var(--accent); }}
.kpi-tint .kpi-label {{ color: var(--txt); }}

/* Tables */
.tbl {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: .92rem;
       border: 1px solid #E6E9F0; border-radius: 12px; overflow: hidden; }}
.tbl th {{ background: #F5F7FB; color: #344054; font-weight: 600; padding: 10px 14px; text-align: right; }}
.tbl td {{ padding: 9px 14px; border-top: 1px solid #EEF0F4; color: #1F2937; }}
.tbl td.num, .tbl th.num {{ text-align: center; direction: ltr; }}
.tbl tr.hl td {{ background: #F0F4FF; font-weight: 600; }}

/* Risk result */
.risk {{ border-radius: 16px; padding: 18px 20px; border: 1px solid var(--accent);
        background: var(--bg); color: var(--txt); display: flex; gap: 14px; align-items: center; }}
.risk-icon {{ font-size: 2rem; }}
.risk-title {{ font-size: 1.3rem; font-weight: 800; }}
.risk-text {{ font-size: .95rem; margin-top: 2px; }}
.score {{ text-align: center; margin: 4px 0 12px 0; }}
.score .val {{ font-size: 3rem; font-weight: 800; line-height: 1.1; color: var(--accent); }}
.score .lbl {{ color: #667085; font-size: .9rem; }}

.note {{ background: #F5F7FB; border-radius: 12px; padding: 12px 16px; color: #475467; font-size: .9rem; }}
.sidebar-brand {{ font-size: 1.15rem; font-weight: 800; color: {NAVY}; margin-bottom: 4px; }}
.tbl-wrap {{ width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch; }}

/* Footer */
.footer {{
  margin-top: 48px; padding: 22px 16px 6px 16px; border-top: 1px solid #E6E9F0;
  text-align: center; color: #667085; font-size: .9rem; letter-spacing: .2px;
}}

/* Tablet */
@media (max-width: 1024px) {{
  .block-container {{ padding-left: 1.5rem; padding-right: 1.5rem; }}
  .kpi-value {{ font-size: 1.7rem; }}
}}

/* Mobile */
@media (max-width: 640px) {{
  .block-container {{ padding: 1.2rem 1rem 2rem 1rem; }}
  [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; gap: .75rem; }}
  [data-testid="stColumn"] {{ width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }}
  [data-testid="stColumn"]:empty, [data-testid="stColumn"]:not(:has(*)) {{ display: none; }}

  .hero {{ padding: 18px 18px; border-radius: 14px; margin-bottom: 12px; }}
  .hero h1 {{ font-size: 1.35rem; line-height: 1.35; }}
  .hero p {{ font-size: .9rem; }}
  .hero .chip {{ font-size: .75rem; padding: 3px 10px; }}

  .section-title {{ font-size: 1.05rem; margin: 16px 0 8px 0; }}
  .section-sub {{ font-size: .82rem; }}

  .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
  .kpi {{ padding: 12px 12px; border-radius: 12px; }}
  .kpi-label {{ font-size: .8rem; }}
  .kpi-value {{ font-size: 1.45rem; }}
  .kpi-sub {{ font-size: .72rem; }}

  .tbl {{ font-size: .8rem; }}
  .tbl th, .tbl td {{ padding: 7px 9px; }}

  .stTabs [data-baseweb="tab-list"] {{ overflow-x: auto; flex-wrap: nowrap; }}
  .stTabs [data-baseweb="tab"] {{ padding: 8px 10px; font-size: .88rem; white-space: nowrap; }}

  .score .val {{ font-size: 2.4rem; }}
  .risk {{ padding: 14px; gap: 10px; }}
  .risk-icon {{ font-size: 1.6rem; }}
  .risk-title {{ font-size: 1.08rem; }}
  .risk-text {{ font-size: .88rem; }}

  .stButton button, .stFormSubmitButton button {{ min-height: 44px; }}
  .footer {{ margin-top: 32px; font-size: .82rem; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Streamlit's slider takes its direction from the browser locale: in a Hebrew-locale browser the
# thumb is positioned right-to-left while the track and tick labels stay left-to-right, so the
# thumb ends up mirrored. Pinning the page's reported locale to en-US keeps sliders consistent.
st.iframe(
    """<script>
    const w = window.parent;
    if (w.navigator.language !== "en-US") {
      Object.defineProperty(w.navigator, "language", {get: () => "en-US", configurable: true});
      Object.defineProperty(w.navigator, "languages", {get: () => ["en-US", "en"], configurable: true});
      w.dispatchEvent(new Event("languagechange"));
    }
    </script>""",
    height=1,
)


# ---------------------------------------------------------------- helpers
def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    html(f'<div class="section-title">{title}</div>' + (f'<div class="section-sub">{sub}</div>' if sub else ""))


def kpi(label: str, value: str, sub: str = "", accent: str = BLUE, tint: tuple | None = None) -> str:
    style = f"--accent:{accent};"
    cls = "kpi"
    if tint:
        cls += " kpi-tint"
        style += f"--bg:{tint[0]};--txt:{tint[1]};"
    return (f'<div class="{cls}" style="{style}"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>')


def kpi_grid(cards: list[str]) -> None:
    html(f'<div class="kpi-grid">{"".join(cards)}</div>')


def table(headers: list[str], rows: list[list[str]], num_cols: tuple = (), highlight: int | None = None) -> None:
    head = "".join(f'<th class="{"num" if i in num_cols else ""}">{h}</th>' for i, h in enumerate(headers))
    body = ""
    for r_i, row in enumerate(rows):
        cells = "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(row))
        body += f'<tr class="{"hl" if r_i == highlight else ""}">{cells}</tr>'
    html(f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family="Heebo, sans-serif", size=13, color="#1F2937"),
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0.5, xanchor="center"),
    )
    return fig


def chart(fig: go.Figure, height: int = 380) -> None:
    st.plotly_chart(style_fig(fig, height), width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------- data & model
@st.cache_resource(show_spinner="טוען את המודל...")
def get_artifacts() -> dict:
    """Load the trained model; train and save it if no compatible artifact exists."""
    if ARTIFACT_PATH.exists():
        try:
            return load_artifacts()
        except Exception:
            pass  # missing/incompatible artifact (e.g. other sklearn version on the cloud) -> retrain
    artifacts = train_and_evaluate()
    save_artifacts(artifacts)
    return artifacts


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data()


artifacts = get_artifacts()
df = get_data()
summary = dataset_summary(df)
pipeline = artifacts["pipeline"]
tuned_threshold = float(round(artifacts["tuning"]["threshold"], 3))
default_metrics = artifacts["metrics_default"]

st.session_state.setdefault("threshold", tuned_threshold)
for key, value in INPUT_DEFAULTS.items():
    st.session_state.setdefault(key, value)


def _reset_threshold() -> None:
    st.session_state["threshold"] = tuned_threshold


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    html('<div class="sidebar-brand">💳 הגדרות המודל</div>')
    st.caption("סף ההחלטה קובע מאיזה ציון סיכון עסקה מסווגת כהונאה.")
    threshold = st.slider(
        "סף החלטה (Decision Threshold)",
        min_value=0.05,
        max_value=0.99,
        step=0.001,
        format="%.3f",
        key="threshold",
        help="עסקה שציון הסיכון שלה גבוה מהסף או שווה לו תסווג כהונאה. "
             "סף נמוך יותר מזהה יותר הונאות, אך מייצר יותר התראות שווא.",
    )
    st.button("↺ חזרה לסף המכויל", on_click=_reset_threshold, width="stretch",
              disabled=abs(threshold - tuned_threshold) < 1e-9)
    html(
        f'<div class="note">הסף המכויל <b>{tuned_threshold:.3f}</b> נבחר באמצעות '
        f'<bdi>{CV_FOLDS}-fold Stratified Cross-Validation</bdi> על סט האימון, כך שימקסם את ה-<bdi>F1-Score</bdi>.</div>'
    )

    st.divider()
    html('<div class="sidebar-brand">ℹ️ פרטי המודל</div>')
    table(
        ["פרמטר", "ערך"],
        [
            ["אלגוריתם", "Logistic Regression"],
            ["טיפול בחוסר איזון", "<bdi>class_weight='balanced'</bdi>"],
            ["עסקאות באימון", f"{artifacts['n_train']:,}"],
            ["עסקאות בבדיקה", f"{artifacts['n_test']:,}"],
            ["הונאות בבדיקה", f"{artifacts['n_test_fraud']}"],
        ],
    )
    st.write("")
    if st.button("🔄 אימון מחדש של המודל", width="stretch"):
        get_artifacts.clear()
        if ARTIFACT_PATH.exists():
            ARTIFACT_PATH.unlink()
        st.rerun()

metrics = evaluate(artifacts["y_test"], artifacts["test_proba"], threshold)

# ---------------------------------------------------------------- header
html(
    '<div class="hero"><h1>מערכת לזיהוי הונאות בכרטיסי אשראי</h1>'
    '<p>מודל Logistic Regression עם שקלול מחלקות וסף החלטה מכויל, לניתוח ביצועים ולהערכת סיכון בזמן אמת</p>'
    '<div class="chips">'
    f'<span class="chip">{summary["n_rows"]:,} עסקאות</span>'
    f'<span class="chip">{summary["fraud_rate"]:.2%} הונאות</span>'
    f'<span class="chip">ROC AUC {metrics["roc_auc"]:.3f}</span>'
    f'<span class="chip">סף נוכחי {threshold:.3f}</span>'
    "</div></div>"
)

tab_perf, tab_predict = st.tabs(["📊  ביצועי המודל וניתוח", "🔍  חיזוי הונאה אינטראקטיבי"])

# ================================================================ tab 1
with tab_perf:
    section("מדדי ביצוע מרכזיים", f"הערכה על סט הבדיקה ({artifacts['n_test']:,} עסקאות) בסף {threshold:.3f}")
    kpi_grid([
        kpi("Precision", f"{metrics['precision']:.3f}", f"שיעור ההתראות שהן הונאה אמיתית · בסף 0.5: {default_metrics['precision']:.3f}"),
        kpi("Recall", f"{metrics['recall']:.3f}", f"שיעור ההונאות שזוהו · בסף 0.5: {default_metrics['recall']:.3f}"),
        kpi("F1-Score", f"{metrics['f1']:.3f}", f"איזון בין Precision ל-Recall · בסף 0.5: {default_metrics['f1']:.3f}"),
        kpi("ROC AUC", f"{metrics['roc_auc']:.3f}", "יכולת דירוג כוללת · אינו תלוי בסף", accent=NAVY),
        kpi("PR AUC", f"{metrics['pr_auc']:.3f}", "איכות הדירוג במחלקת ההונאה · אינו תלוי בסף", accent=NAVY),
    ])

    tn, fp, fn, tp = metrics["confusion_matrix"].ravel()
    section("השפעה עסקית", "תוצאות הסיווג על סט הבדיקה במונחים עסקיים")
    kpi_grid([
        kpi("הונאות שנתפסו", f"{tp} מתוך {tp + fn}", "עסקאות הונאה שנחסמו", GREEN, (GREEN_BG, GREEN_TXT)),
        kpi("הונאות שהוחמצו", f"{fn}", "False Negatives · הפסד כספי פוטנציאלי", RED, (RED_BG, RED_TXT)),
        kpi("התראות שווא", f"{fp}", "False Positives · לקוחות לגיטימיים שנחסמו", AMBER, (AMBER_BG, AMBER_TXT)),
        kpi("עסקאות תקינות שאושרו", f"{tn:,}", "True Negatives", BLUE, ("#EEF2FF", "#2B3F7A")),
    ])
    st.caption(
        f"סט הבדיקה כולל {artifacts['n_test_fraud']} מקרי הונאה בלבד, ולכן כל טעות סיווג בודדת "
        f"משנה את ה-Recall בכ-{100 / artifacts['n_test_fraud']:.1f} נקודות."
    )

    section("ניתוח גרפי")
    col_cm, col_curves = st.columns([1, 1.35], gap="large")

    with col_cm:
        with st.container(border=True):
            html("<b>מטריצת בלבול (Confusion Matrix)</b>")
            normalize = st.toggle("הצגה באחוזים (נרמול לפי המחלקה בפועל)", value=False)
            cm = metrics["confusion_matrix_norm"] if normalize else metrics["confusion_matrix"]
            labels = ["לגיטימי", "הונאה"]
            fig = px.imshow(
                cm, x=labels, y=labels, text_auto=".1%" if normalize else "d",
                color_continuous_scale=["#F5F7FB", "#AFC0F5", BLUE, NAVY],
                labels={"x": "סיווג המודל", "y": "בפועל", "color": ""},
            )
            fig.update_traces(textfont_size=20)
            fig.update_layout(coloraxis_showscale=False)
            chart(fig, 390)

    with col_curves:
        with st.container(border=True):
            roc_tab, pr_tab = st.tabs(["עקומת ROC", "עקומת Precision-Recall"])
            with roc_tab:
                fpr, tpr = metrics["roc_curve"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"המודל (AUC = {metrics['roc_auc']:.3f})",
                                         line=dict(color=BLUE, width=3), fill="tozeroy", fillcolor="rgba(59,91,219,.08)"))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="ניחוש אקראי",
                                         line=dict(dash="dash", color=GRAY)))
                fig.add_trace(go.Scatter(x=[fp / (fp + tn)], y=[tp / (tp + fn)], mode="markers", name="נקודת העבודה הנוכחית",
                                         marker=dict(size=14, color=RED, line=dict(color="white", width=2))))
                fig.update_layout(xaxis_title="שיעור התראות שווא (FPR)", yaxis_title="שיעור זיהוי (TPR)")
                chart(fig, 410)
            with pr_tab:
                prec, rec = metrics["pr_curve"]
                baseline = artifacts["n_test_fraud"] / artifacts["n_test"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"המודל (AP = {metrics['pr_auc']:.3f})",
                                         line=dict(color=BLUE, width=3)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[baseline, baseline], mode="lines",
                                         name=f"קו בסיס ({baseline:.1%})", line=dict(dash="dash", color=GRAY)))
                fig.add_trace(go.Scatter(x=[metrics["recall"]], y=[metrics["precision"]], mode="markers",
                                         name="נקודת העבודה הנוכחית",
                                         marker=dict(size=14, color=RED, line=dict(color="white", width=2))))
                fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
                chart(fig, 410)

    col_coef, col_data = st.columns([1.2, 1], gap="large")

    with col_coef:
        section("גורמי הסיכון במודל", "מקדמי המודל על משתנים מנורמלים: אדום מעלה את הסיכון, ירוק מוריד אותו")
        coefs = artifacts["coefficients"].iloc[::-1].copy()
        coefs["label"] = coefs["feature"].map(feature_label)
        fig = go.Figure(go.Bar(
            x=coefs["coefficient"], y=coefs["label"], orientation="h",
            marker_color=np.where(coefs["coefficient"] > 0, RED, GREEN),
            text=coefs["coefficient"].round(2), textposition="outside", cliponaxis=False,
        ))
        fig.update_layout(xaxis_title="מקדם (Coefficient)", showlegend=False)
        fig.update_xaxes(range=[coefs["coefficient"].min() * 1.3 - 0.3, coefs["coefficient"].max() * 1.15 + 0.3])
        chart(fig, 470)

    with col_data:
        section("סטטיסטיקות הנתונים")
        kpi_grid([
            kpi("סך העסקאות", f"{summary['n_rows']:,}"),
            kpi("עסקאות הונאה", f"{summary['n_fraud']}", f"{summary['fraud_rate']:.2%} מהעסקאות", RED),
            kpi("יחס חוסר איזון", f"1:{summary['imbalance_ratio']:.0f}", "הונאה מול לגיטימי", AMBER),
        ])
        st.write("")
        means = summary["class_means"]
        rows = []
        for feat in means.columns:
            legit, fraud = means.loc[0, feat], means.loc[1, feat]
            if feat in BINARY_FEATURES:
                rows.append([FEATURE_HE[feat], f"{legit:.0%}", f"{fraud:.0%}"])
            else:
                rows.append([FEATURE_HE[feat], f"{legit:,.1f}", f"{fraud:,.1f}"])
        table(["משתנה", "ממוצע: לגיטימי", "ממוצע: הונאה"], rows, num_cols=(1, 2))

    section("מתודולוגיה וטיפול בחוסר האיזון")
    m1, m2 = st.columns([1.2, 1], gap="large")
    with m1:
        with st.container(border=True):
            html(
                f"""<p><b>1. שקלול מחלקות:</b> ההגדרה <bdi><code>class_weight='balanced'</code></bdi> גורמת לכך שטעות בעסקת הונאה
"עולה" למודל כ-{summary['imbalance_ratio']:.0f} פעמים יותר מטעות בעסקה לגיטימית. כך המודל לומד מכל
{artifacts['n_train']:,} עסקאות האימון, ללא יצירת נתונים סינתטיים.</p>
<p><b>2. כיול סף ההחלטה:</b> שקלול המחלקות מנפח את ציוני הסיכון, ולכן סף ברירת המחדל 0.5 מייצר התראות שווא רבות.
באמצעות <bdi>{CV_FOLDS}-fold Stratified Cross-Validation</bdi> על סט האימון נבחר הסף שממקסם את ה-<bdi>F1-Score</bdi>:
<b>{tuned_threshold:.3f}</b>. סט הבדיקה לא שימש לכיול.</p>
<p><b>3. עיבוד מקדים:</b> נרמול (<bdi>StandardScaler</bdi>) למשתנים רציפים, One-Hot Encoding לקטגוריית בית העסק,
והסרת <bdi><code>transaction_id</code></bdi>. החלוקה לאימון ולבדיקה היא מרובדת (Stratified), ביחס 80/20.</p>"""
            )
    with m2:
        tuned_m = artifacts["metrics"]
        cols = ["מדד", "סף 0.5", f"סף מכויל ({tuned_threshold:.3f})"]
        rows = [
            ["Precision", f"{default_metrics['precision']:.3f}", f"{tuned_m['precision']:.3f}"],
            ["Recall", f"{default_metrics['recall']:.3f}", f"{tuned_m['recall']:.3f}"],
            ["F1-Score", f"{default_metrics['f1']:.3f}", f"{tuned_m['f1']:.3f}"],
            ["התראות שווא", f"{default_metrics['confusion_matrix'][0, 1]}", f"{tuned_m['confusion_matrix'][0, 1]}"],
            ["הונאות שהוחמצו", f"{default_metrics['confusion_matrix'][1, 0]}", f"{tuned_m['confusion_matrix'][1, 0]}"],
        ]
        if abs(threshold - tuned_threshold) > 1e-9:
            cols.append(f"סף נוכחי ({threshold:.3f})")
            extra = [f"{metrics['precision']:.3f}", f"{metrics['recall']:.3f}", f"{metrics['f1']:.3f}", f"{fp}", f"{fn}"]
            rows = [r + [e] for r, e in zip(rows, extra)]
        html("<b>השפעת כיול הסף על סט הבדיקה</b>")
        table(cols, rows, num_cols=tuple(range(1, len(cols))), highlight=2)
        st.write("")
        html(
            '<div class="note">בשל שקלול המחלקות, ציון הסיכון אינו הסתברות מכוילת: הוא מגזים בסבירות '
            "האמיתית להונאה (שיעור הבסיס הוא כ-1.5%). יש להשוות אותו לסף ההחלטה.</div>"
        )


# ================================================================ tab 2
def _set_inputs(values: dict) -> None:
    for k, v in values.items():
        st.session_state[k] = v


def _sample_from_test(label: int) -> dict:
    """Random test transaction of the given class that the model classifies correctly."""
    X_test, y_test, proba = artifacts["X_test"], artifacts["y_test"], artifacts["test_proba"]
    correct = (proba >= artifacts["tuning"]["threshold"]) == bool(label)
    row = X_test.iloc[np.random.choice(np.flatnonzero((y_test == label) & correct))]
    return {
        "amount": float(row["amount"]),
        "transaction_hour": int(row["transaction_hour"]),
        "merchant_category": row["merchant_category"],
        "foreign_transaction": bool(row["foreign_transaction"]),
        "location_mismatch": bool(row["location_mismatch"]),
        "device_trust_score": int(row["device_trust_score"]),
        "velocity_last_24h": int(row["velocity_last_24h"]),
        "cardholder_age": int(row["cardholder_age"]),
    }


def _load_sample(label: int) -> None:
    values = _sample_from_test(label)
    _set_inputs(values)
    st.session_state["predict_features"] = values
    st.session_state["sample_note"] = (
        "נטענה עסקת הונאה אמיתית מסט הבדיקה" if label else "נטענה עסקה לגיטימית אמיתית מסט הבדיקה"
    )


def _reset_inputs() -> None:
    _set_inputs(INPUT_DEFAULTS)
    st.session_state.pop("predict_features", None)
    st.session_state.pop("sample_note", None)


def risk_level(p: float, thr: float) -> dict:
    if p >= thr:
        return dict(accent=RED, bg=RED_BG, txt=RED_TXT, icon="🚨", title="סיכון גבוה: חשד להונאה",
                    text="העסקה מסווגת כהונאה. מומלץ לחסום אותה ולאמת מול בעל הכרטיס.")
    if p >= thr / 2:
        return dict(accent=AMBER, bg=AMBER_BG, txt=AMBER_TXT, icon="⚠️", title="סיכון בינוני",
                    text="העסקה מסווגת כלגיטימית, אך ציון הסיכון קרוב לסף. מומלץ ניטור.")
    return dict(accent=GREEN, bg=GREEN_BG, txt=GREEN_TXT, icon="✅", title="סיכון נמוך: עסקה לגיטימית",
                text="לא זוהו סימנים מחשידים משמעותיים. ניתן לאשר את העסקה.")


with tab_predict:
    section("בדיקת עסקה", "הזינו את פרטי העסקה או טענו דוגמה אמיתית מסט הבדיקה, ולחצו על 'חשב סיכון הונאה'")
    b1, b2, b3, _ = st.columns([1, 1, 0.7, 1.6])
    b1.button("🚨 טען דוגמת הונאה", on_click=_load_sample, args=(1,), width="stretch")
    b2.button("✅ טען דוגמה לגיטימית", on_click=_load_sample, args=(0,), width="stretch")
    b3.button("↺ איפוס", on_click=_reset_inputs, width="stretch")
    if note := st.session_state.get("sample_note"):
        st.caption(f"ℹ️ {note}")

    col_in, col_out = st.columns([1.1, 1], gap="large")

    with col_in:
        with st.form("predict_form", border=True):
            html("<b>💳 פרטי העסקה</b>")
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("סכום העסקה ($)", min_value=0.0, max_value=5000.0, step=10.0, key="amount")
            with c2:
                st.selectbox("קטגוריית בית העסק", list(MERCHANT_HE), format_func=MERCHANT_HE.get,
                             key="merchant_category")
            st.slider("שעת העסקה", 0, 23, key="transaction_hour", format="%d:00",
                      help="הונאות מתרכזות בשעות הלילה המאוחרות")

            st.divider()
            html("<b>🛡️ אותות סיכון</b>")
            c3, c4 = st.columns(2)
            with c3:
                st.toggle('עסקה בחו"ל', key="foreign_transaction")
            with c4:
                st.toggle("אי-התאמת מיקום", key="location_mismatch",
                          help="מיקום העסקה אינו תואם את המיקום הרגיל של בעל הכרטיס")
            st.slider("ציון אמינות המכשיר", 0, 99, key="device_trust_score",
                      help="ציון נמוך מעיד על מכשיר לא מוכר או חשוד")
            st.slider("מספר עסקאות ב-24 השעות האחרונות", 0, 15, key="velocity_last_24h",
                      help="ריבוי עסקאות בזמן קצר הוא סימן מחשיד")

            st.divider()
            html("<b>👤 בעל הכרטיס</b>")
            st.slider("גיל בעל הכרטיס", 18, 90, key="cardholder_age")

            submitted = st.form_submit_button("🔮 חשב סיכון הונאה", type="primary", width="stretch")

    if submitted:
        st.session_state["predict_features"] = {k: st.session_state[k] for k in INPUT_DEFAULTS}
        st.session_state.pop("sample_note", None)

    with col_out:
        features = st.session_state.get("predict_features")
        if features is None:
            with st.container(border=True):
                html(
                    '<div style="text-align:center;padding:60px 10px;color:#667085;">'
                    '<div style="font-size:3rem;">🔍</div>'
                    '<div style="font-size:1.1rem;font-weight:600;color:#344054;margin-top:8px;">טרם בוצעה בדיקה</div>'
                    "<div style='margin-top:6px;'>מלאו את פרטי העסקה ולחצו על 'חשב סיכון הונאה',<br>"
                    "או טענו דוגמה מוכנה.</div></div>"
                )
        else:
            model_input = {**features,
                           "foreign_transaction": int(features["foreign_transaction"]),
                           "location_mismatch": int(features["location_mismatch"])}
            proba = predict_proba(pipeline, model_input)
            lvl = risk_level(proba, threshold)

            with st.container(border=True):
                html(
                    f'<div class="score" style="--accent:{lvl["accent"]};">'
                    f'<div class="lbl">ציון סיכון הונאה</div><div class="val">{proba:.1%}</div></div>'
                    f'<div class="risk" style="--accent:{lvl["accent"]};--bg:{lvl["bg"]};--txt:{lvl["txt"]};">'
                    f'<div class="risk-icon">{lvl["icon"]}</div><div><div class="risk-title">{lvl["title"]}</div>'
                    f'<div class="risk-text">{lvl["text"]}</div></div></div>'
                )
                gauge = go.Figure(go.Indicator(
                    mode="gauge",
                    value=proba * 100,
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%"},
                        "bar": {"color": lvl["accent"], "thickness": 0.3},
                        "steps": [
                            {"range": [0, threshold * 50], "color": GREEN_BG},
                            {"range": [threshold * 50, threshold * 100], "color": AMBER_BG},
                            {"range": [threshold * 100, 100], "color": RED_BG},
                        ],
                        "threshold": {"line": {"color": NAVY, "width": 4}, "thickness": 0.9,
                                      "value": threshold * 100},
                    },
                ))
                gauge.update_layout(margin=dict(l=40, r=40, t=30, b=10))
                st.plotly_chart(gauge.update_layout(height=230, paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Heebo, sans-serif", size=13)),
                                width="stretch", config={"displayModeBar": False})
                st.caption(f"הקו הכהה במד מסמן את סף ההחלטה ({threshold:.1%}).")

            with st.container(border=True):
                html("<b>מה השפיע על הציון?</b>")
                contrib = explain(pipeline, model_input).head(6).iloc[::-1]
                fig = go.Figure(go.Bar(
                    x=contrib["contribution"], y=contrib["feature"].map(feature_label), orientation="h",
                    marker_color=np.where(contrib["contribution"] > 0, RED, GREEN),
                    text=contrib["contribution"].round(2), textposition="outside", cliponaxis=False,
                ))
                lim = max(contrib["contribution"].abs().max() * 1.35, 0.5)
                fig.update_layout(xaxis_title="השפעה על ציון הסיכון (log-odds)", showlegend=False)
                fig.update_xaxes(range=[-lim, lim])
                chart(fig, 300)
                st.caption("אדום: מעלה את הסיכון · ירוק: מוריד את הסיכון (ביחס לעסקה ממוצעת).")

# ---------------------------------------------------------------- footer
html('<div class="footer">© כל הזכויות שמורות לאיתי בסטקר</div>')
