import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
FASTAPI_URL = "https://ibm-hr-midterm-project.onrender.com/predict"  # change if needed

st.set_page_config(page_title="Employee Attrition — Predict & Visualize",
                   layout="wide",
                   initial_sidebar_state="collapsed")

# ---------- CUSTOM CSS ----------
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: linear-gradient(180deg, #f6f8fb 0%, #ffffff 100%);
    }

    /* Card style */
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(32,33,36,0.08);
        margin-bottom: 16px;
    }

    /* Headings */
    .title {
        font-size:28px;
        font-weight:700;
        color:#0f172a;
        margin-bottom:6px;
    }
    .subtitle {
        color:#475569;
        margin-top: -6px;
        margin-bottom: 12px;
        font-size:14px;
    }

    /* Button */
    .stButton>button {
        background: linear-gradient(90deg,#7c3aed,#06b6d4);
        color: white;
        border: none;
        padding: 8px 14px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        filter: brightness(1.03);
    }

    /* Metric style tweak */
    .stMetric > div {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- HEADER ----------
st.markdown('<div class="card"><div class="title">🏢 Employee Attrition Predictor</div>'
            '<div class="subtitle">Fill employee details (left) and click Predict. See charts & summary on the right.</div></div>',
            unsafe_allow_html=True)

# ---------- LAYOUT ----------
left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    with st.form("prediction_form", clear_on_submit=False):
        # group inputs in compact form
        st.markdown('<div class="card"> <strong>Employee information</strong></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
            businesstravel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            dailyrate = st.number_input("Daily Rate", min_value=0, max_value=20000, value=500)
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            distancefromhome = st.number_input("Distance From Home", min_value=0, max_value=500, value=10)
            education = st.selectbox("Education", ["Below College", "College", "Bachelor", "Master", "Doctor"])
            educationfield = st.selectbox("Education Field",
                                          ["Life Sciences", "Medical", "Marketing", "Technical Degree",
                                           "Human Resources", "Other"])
            environmentsatisfaction = st.selectbox("Environment Satisfaction", ["Low", "Medium", "High", "Very High"])

        with c2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            hourlyrate = st.number_input("Hourly Rate", min_value=0, max_value=1000, value=50)
            jobinvolvement = st.selectbox("Job Involvement", ["Low", "Medium", "High", "Very High"])
            joblevel = st.selectbox("Job Level", ["Entry Level", "Junior Level", "Mid Level", "Senior Level", "Executive Level"])
            jobrole = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician",
                                               "Manufacturing Director", "Healthcare Representative", "Manager",
                                               "Sales Representative", "Research Director", "Human Resources"])
            jobsatisfaction = st.selectbox("Job Satisfaction", ["Low", "Medium", "High", "Very High"])
            maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            monthlyincome = st.number_input("Monthly Income", min_value=0, max_value=500000, value=5000)

        # more numeric fields in a compact row
        st.markdown("**More details**")
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            monthlyrate = st.number_input("Monthly Rate", min_value=0, max_value=500000, value=15000)
        with t2:
            numcompaniesworked = st.number_input("Num Companies Worked", min_value=0, max_value=50, value=1)
        with t3:
            overtime = st.selectbox("OverTime", ["Yes", "No"])
        with t4:
            percentsalaryhike = st.number_input("Percent Salary Hike", min_value=0, max_value=100, value=11)

        u1, u2, u3, u4 = st.columns(4)
        with u1:
            performancerating = st.selectbox("Performance Rating", ["Low", "Good", "Excellent", "Outstanding"])
        with u2:
            relationshipsatisfaction = st.selectbox("Relationship Satisfaction", ["Low", "Medium", "High", "Very High"])
        with u3:
            stockoptionlevel = st.number_input("Stock Option Level", min_value=0, max_value=5, value=0)
        with u4:
            totalworkingyears = st.number_input("Total Working Years", min_value=0, max_value=60, value=5)

        v1, v2, v3 = st.columns(3)
        with v1:
            trainingtimeslastyear = st.number_input("Training Times Last Year", min_value=0, max_value=100, value=2)
        with v2:
            worklifebalance = st.selectbox("Work Life Balance", ["Bad", "Good", "Better", "Best"])
        with v3:
            yearsatcompany = st.number_input("Years at Company", min_value=0, max_value=60, value=2)

        w1, w2, w3 = st.columns(3)
        with w1:
            yearsincurrentrole = st.number_input("Years in Current Role", min_value=0, max_value=60, value=1)
        with w2:
            yearssincelastpromotion = st.number_input("Years since Last Promotion", min_value=0, max_value=60, value=0)
        with w3:
            yearswithcurrmanager = st.number_input("Years with Current Manager", min_value=0, max_value=60, value=1)

        submit_btn = st.form_submit_button("Predict")

with right_col:
    # result card
    result_card = st.empty()
    charts_card = st.empty()

# ---------- PREDICTION & VISUALS ----------
if 'submit_btn' not in st.session_state:
    st.session_state['submit_btn'] = False

if submit_btn:
    payload = {
        "age": int(age),
        "businesstravel": businesstravel,
        "dailyrate": int(dailyrate),
        "department": department,
        "distancefromhome": int(distancefromhome),
        "education": str(education),
        "educationfield": educationfield,
        "environmentsatisfaction": str(environmentsatisfaction),
        "gender": gender,
        "hourlyrate": int(hourlyrate),
        "jobinvolvement": str(jobinvolvement),
        "joblevel": str(joblevel),
        "jobrole": jobrole,
        "jobsatisfaction": str(jobsatisfaction),
        "maritalstatus": maritalstatus,
        "monthlyincome": int(monthlyincome),
        "monthlyrate": int(monthlyrate),
        "numcompaniesworked": int(numcompaniesworked),
        "overtime": overtime,
        "percentsalaryhike": int(percentsalaryhike),
        "performancerating": str(performancerating),
        "relationshipsatisfaction": str(relationshipsatisfaction),
        "stockoptionlevel": int(stockoptionlevel),
        "totalworkingyears": int(totalworkingyears),
        "trainingtimeslastyear": int(trainingtimeslastyear),
        "worklifebalance": str(worklifebalance),
        "yearsatcompany": int(yearsatcompany),
        "yearsincurrentrole": int(yearsincurrentrole),
        "yearssincelastpromotion": int(yearssincelastpromotion),
        "yearswithcurrmanager": int(yearswithcurrmanager),
    }

  
    try:
        with st.spinner("Sending data to model..."):
            resp = requests.post(FASTAPI_URL, json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
    except requests.exceptions.RequestException as e:
        result_card.markdown(
            '<div class="card"><strong style="color:#b91c1c">Error</strong><div style="color:#475569">'
            f'Could not connect to backend: {str(e)}'
            '</div></div>',
            unsafe_allow_html=True)
    else:
        # extract
        pred_label = result.get("prediction", None)
        prob = result.get("probability_of_leaving", None)
        message = result.get("message", "")

        # RESULT CARD (top)
        prob_pct = None
        try:
            prob_pct = float(prob)
        except Exception:
            prob_pct = None

        # human friendly
        status_color = "#ef4444" if pred_label == 1 else "#059669"
        status_text = "AT RISK — MAY LEAVE" if pred_label == 1 else "STABLE — LIKELY TO STAY"

        # render result card
        result_card.markdown(
            f'''
            <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h2 style="margin:0;">Prediction</h2>
                        <div style="font-size:16px; color:#475569; margin-top:4px">{message}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-weight:700; font-size:18px; color:{status_color};">{status_text}</div>
                        <div style="color:#94a3b8; font-size:12px;">Model label: {pred_label}</div>
                    </div>
                </div>
                <hr style="margin:12px 0;">
                <div style="display:flex; gap:16px; align-items:center;">
                    <div style="flex:1;">
                        <div style="font-size:13px; color:#64748b;">Probability of leaving</div>
                        <div style="font-weight:700; font-size:20px;">{prob_pct:.4f}</div>
                        <div style="margin-top:6px;">
                            <!-- progress bar -->
                            <progress value="{prob_pct if prob_pct is not None else 0}" max="1" style="width:100%; height:14px;"></progress>
                        </div>
                    </div>
                    <div style="width:1px; background:#eef2ff; height:64px;"></div>
                    <div style="min-width:220px;">
                        <div style="font-size:13px; color:#64748b;">Quick metrics</div>
                        <div style="display:flex; gap:8px; margin-top:8px;">
                            <div style="background:#f8fafc; padding:8px; border-radius:8px;">
                                <div style="font-size:12px; color:#475569">Age</div>
                                <div style="font-weight:700;">{age}</div>
                            </div>
                            <div style="background:#f8fafc; padding:8px; border-radius:8px;">
                                <div style="font-size:12px; color:#475569">Monthly Income</div>
                                <div style="font-weight:700;">{monthlyincome}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

        # CHARTS
        # 1) Bar chart of numeric fields
        numeric_df = pd.DataFrame({
            "feature": ["Age", "Monthly Income", "Total Working Years", "Years at Company"],
            "value": [age, monthlyincome, totalworkingyears, yearsatcompany]
        }).set_index("feature")

        fig1, ax1 = plt.subplots(figsize=(6, 3.6))
        numeric_df["value"].plot(kind="bar", ax=ax1)
        ax1.set_ylabel("Value")
        ax1.set_title("Key numeric attributes (single employee)")
        ax1.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()

        # 2) Donut chart for categorical quick breakdown (overtime, gender, maritalstatus)
        cats = {"OverTime": overtime, "Gender": gender, "Marital": maritalstatus}
        cat_names = list(cats.keys())
        cat_values = [1, 1, 1]  # we display categorical values as equal slices but label them
        fig2, ax2 = plt.subplots(figsize=(4, 3.6))
        wedges, texts = ax2.pie(cat_values, wedgeprops=dict(width=0.5), startangle=-40)
        ax2.legend(wedges, [f"{k}: {v}" for k, v in cats.items()], title="Categories", loc="center left", bbox_to_anchor=(1, 0.5))
        ax2.set_title("Categorical snapshot")
        plt.tight_layout()

        # display charts
        with charts_card.container():
            st.markdown('<div class="card"><strong>Visuals</strong></div>', unsafe_allow_html=True)
            c1, c2 = st.columns([1, 0.7])
            with c1:
                st.pyplot(fig1)
            with c2:
                st.pyplot(fig2)

        # small explanation / tips
        st.markdown(
            """
            <div style="margin-top:8px; color:#475569;">
            <em>Tip:</em> Use the numeric bars to quickly spot unusually high/low values for this employee. 
            If you'd like cohort-level charts (compare to peers), upload a CSV with multiple employees and I can add cohort analytics.
            </div>
            """,
            unsafe_allow_html=True
        )
