
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="AI Student Impact Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# LOAD DATA
# -------------------------------

st.title("📊 AI Student Impact Analytics")
st.caption("Understanding the relationship between Generative AI and student outcomes")

@st.cache_data
def load_data():
    data = pd.read_csv("AI_Student_Impact_Cleaned.csv")

    if "GPA_Change" not in data.columns:
        data["GPA_Change"] = (
            data["Post_Semester_GPA"] -
            data["Pre_Semester_GPA"]
        )

    if "AI_Usage_Category" not in data.columns:
        data["AI_Usage_Category"] = pd.cut(
            data["Weekly_GenAI_Hours"],
            bins=[-0.01, 5, 10, 20, 168],
            labels=["Low", "Moderate", "High", "Very High"]
        )

    return data

df = load_data()

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------

st.sidebar.header("Dashboard Filters")

major_options = sorted(df["Major_Category"].dropna().unique())

selected_majors = st.sidebar.multiselect(
    "Select Major",
    major_options,
    default=major_options
)

year_options = sorted(df["Year_of_Study"].dropna().unique())

selected_years = st.sidebar.multiselect(
    "Year of Study",
    year_options,
    default=year_options
)

usage_options = [
    "Low", "Moderate", "High", "Very High"
]

available_usage = [
    x for x in usage_options
    if x in df["AI_Usage_Category"].astype(str).unique()
]

selected_usage = st.sidebar.multiselect(
    "AI Usage Level",
    available_usage,
    default=available_usage
)

filtered_df = df[
    (df["Major_Category"].isin(selected_majors)) &
    (df["Year_of_Study"].isin(selected_years)) &
    (df["AI_Usage_Category"].astype(str).isin(selected_usage))
]

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# -------------------------------
# KPI CARDS
# -------------------------------

st.subheader("Overall Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Students",
    f"{len(filtered_df):,}"
)

col2.metric(
    "Average AI Hours",
    f"{filtered_df['Weekly_GenAI_Hours'].mean():.2f}"
)

col3.metric(
    "Average Post GPA",
    f"{filtered_df['Post_Semester_GPA'].mean():.2f}"
)

col4.metric(
    "Average Skill Retention",
    f"{filtered_df['Skill_Retention_Score'].mean():.2f}"
)

st.divider()

# -------------------------------
# CHART 1: AI USAGE VS GPA
# -------------------------------

st.subheader("AI Usage and Academic Performance")

usage_gpa = filtered_df.groupby(
    "AI_Usage_Category",
    observed=True
)["GPA_Change"].mean().reindex(
    ["Low", "Moderate", "High", "Very High"]
).dropna().reset_index()

fig1 = px.bar(
    usage_gpa,
    x="AI_Usage_Category",
    y="GPA_Change",
    color="GPA_Change",
    title="Average GPA Change by AI Usage",
    labels={
        "AI_Usage_Category": "AI Usage",
        "GPA_Change": "Average GPA Change"
    }
)

st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# CHART 2: SKILL RETENTION
# -------------------------------

col1, col2 = st.columns(2)

with col1:
    retention = filtered_df.groupby(
        "AI_Usage_Category",
        observed=True
    )["Skill_Retention_Score"].mean().reset_index()

    fig2 = px.bar(
        retention,
        x="AI_Usage_Category",
        y="Skill_Retention_Score",
        title="Skill Retention by AI Usage",
        color="Skill_Retention_Score"
    )

    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# CHART 3: BURNOUT RISK
# -------------------------------

with col2:
    burnout = filtered_df["Burnout_Risk_Level"].value_counts(
    ).reset_index()

    burnout.columns = ["Burnout Risk", "Students"]

    fig3 = px.pie(
        burnout,
        names="Burnout Risk",
        values="Students",
        title="Student Burnout Risk Distribution",
        hole=0.4
    )

    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# CHART 4: MAJOR ANALYSIS
# -------------------------------

st.subheader("AI Usage Across Academic Majors")

major_analysis = filtered_df.groupby(
    "Major_Category"
)["Weekly_GenAI_Hours"].mean().reset_index()

fig4 = px.bar(
    major_analysis,
    x="Major_Category",
    y="Weekly_GenAI_Hours",
    color="Weekly_GenAI_Hours",
    title="Average Weekly AI Usage by Major"
)

st.plotly_chart(fig4, use_container_width=True)

# -------------------------------
# CHART 5: GPA COMPARISON
# -------------------------------

st.subheader("Academic Performance Comparison")

gpa_data = pd.DataFrame({
    "Semester": ["Pre-Semester", "Post-Semester"],
    "Average GPA": [
        filtered_df["Pre_Semester_GPA"].mean(),
        filtered_df["Post_Semester_GPA"].mean()
    ]
})

fig5 = px.bar(
    gpa_data,
    x="Semester",
    y="Average GPA",
    text_auto=".2f",
    title="Average GPA Before and After Semester"
)

st.plotly_chart(fig5, use_container_width=True)

# -------------------------------
# DATA TABLE
# -------------------------------

st.subheader("Student Data Explorer")

st.dataframe(
    filtered_df.head(500),
    use_container_width=True
)

# -------------------------------
# DOWNLOAD FILTERED DATA
# -------------------------------

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Dataset",
    data=csv,
    file_name="Filtered_Student_Analysis.csv",
    mime="text/csv"
)

st.caption(
    "Note: These visualizations show associations in the dataset, "
    "not proof of causal effects."
)
