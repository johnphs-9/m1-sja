import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import duckdb

#con = duckdb.connect("sgJobData.db")

# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Singapore Job Market Recruitment Insights",
    page_icon="💼",
    layout="wide",
)

st.title("Singapore Job Market Recruitment Insights Dashboard")
st.caption(
    "For HR analysts and recruiters: identify high-demand roles, benchmark salary offers, "
    "and prioritise recruitment strategies for roles with many vacancies but low applicant response."
)


# ============================================================
# Helper functions
# ============================================================
def parse_categories(value):
    """Extract category names from the JSON-like categories column."""
    if pd.isna(value):
        return ["Uncategorised"]

    if isinstance(value, list):
        records = value
    else:
        try:
            records = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            try:
                records = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return ["Uncategorised"]

    if not isinstance(records, list) or len(records) == 0:
        return ["Uncategorised"]

    categories = []
    for item in records:
        if isinstance(item, dict):
            category = item.get("category")
            if category:
                categories.append(category)

    return categories if categories else ["Uncategorised"]


@st.cache_data(show_spinner=False)
def load_data_from_path(path):
    """Load and prepare the job-postings dataset from a local CSV path."""
    data = pd.read_csv(path)
    return prepare_data(data)


@st.cache_data(show_spinner=False)
def load_data_from_upload(uploaded_file):
    """Load and prepare the job-postings dataset from an uploaded CSV file."""
    data = pd.read_csv(uploaded_file)
    return prepare_data(data)


def prepare_data(data):
    """Clean and enrich the dataset for dashboard analysis."""
    df = data.copy()

    # Dates
    df["posting_date"] = pd.to_datetime(df["metadata_newPostingDate"], errors="coerce")
    df["posting_year"] = df["posting_date"].dt.year

    # Text columns
    for col in ["employmentTypes", "positionLevels", "title", "status_jobStatus"]:
        if col in df.columns:
            df[col] = df[col].fillna("Missing")

    # Numeric columns
    numeric_cols = [
        "average_salary",
        "salary_minimum",
        "salary_maximum",
        "minimumYearsExperience",
        "numberOfVacancies",
        "metadata_totalNumberJobApplication",
        "metadata_totalNumberOfView",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Category parsing and explosion: one row per posting-category pair
    df["category_list"] = df["categories"].apply(parse_categories)
    df_categories = df.explode("category_list").rename(columns={"category_list": "category"})

    # Derived metrics
    df_categories["applications_per_vacancy"] = np.where(
        df_categories["numberOfVacancies"] > 0,
        df_categories["metadata_totalNumberJobApplication"] / df_categories["numberOfVacancies"],
        np.nan,
    )
    df_categories["view_to_application_rate"] = np.where(
        df_categories["metadata_totalNumberOfView"] > 0,
        df_categories["metadata_totalNumberJobApplication"] / df_categories["metadata_totalNumberOfView"],
        np.nan,
    )

    # A cleaned salary field for salary benchmarking
    df_categories["average_salary_clean"] = df_categories["average_salary"].where(
        df_categories["average_salary"] > 0
    )

    return df_categories


def format_currency(value):
    if pd.isna(value):
        return "N/A"
    return f"S${value:,.0f}"


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else np.nan


def make_summary(data, group_col):
    """Aggregate dashboard metrics by category or role title."""
    summary = (
        data.groupby(group_col, dropna=False)
        .agg(
            postings=("metadata_jobPostId", "nunique"),
            vacancies=("numberOfVacancies", "sum"),
            average_salary=("average_salary_clean", "mean"),
            median_salary=("average_salary_clean", "median"),
            avg_experience=("minimumYearsExperience", "mean"),
            total_views=("metadata_totalNumberOfView", "sum"),
            total_applications=("metadata_totalNumberJobApplication", "sum"),
        )
        .reset_index()
    )

    summary["applications_per_vacancy"] = np.where(
        summary["vacancies"] > 0,
        summary["total_applications"] / summary["vacancies"],
        np.nan,
    )
    summary["view_to_application_rate"] = np.where(
        summary["total_views"] > 0,
        summary["total_applications"] / summary["total_views"],
        np.nan,
    )
    return summary


def add_hard_to_fill_flag(summary, vacancy_quantile=0.75, application_quantile=0.25):
    """Flag groups with high vacancies but low applications per vacancy."""
    result = summary.copy()
    vacancy_threshold = result["vacancies"].quantile(vacancy_quantile)
    app_threshold = result["applications_per_vacancy"].quantile(application_quantile)

    result["hard_to_fill"] = (
        (result["vacancies"] >= vacancy_threshold)
        & (result["applications_per_vacancy"] <= app_threshold)
    )
    return result, vacancy_threshold, app_threshold


# ============================================================
# Data source
# ============================================================
st.sidebar.header("Data source")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

default_candidates = [
    Path("data/SGJobData_10000_random.csv"),
    Path("data/SGJobData.csv"),
]

default_path = next((p for p in default_candidates if p.exists()), None)

if uploaded_file is not None:
    df = load_data_from_upload(uploaded_file)
    data_source_label = uploaded_file.name
elif default_path is not None:
    df = load_data_from_path(default_path)
    data_source_label = default_path.name
else:
    st.error(
        "No dataset found. Upload a CSV file in the sidebar, or place "
        "data/SGJobData_10000_random.csv / data/SGJobData.csv in the same folder as app.py."
    )
    st.stop()

st.sidebar.success(f"Using: {data_source_label}")


# ============================================================
# Sidebar filters
# ============================================================
st.sidebar.header("Filters")

available_years = sorted(df["posting_year"].dropna().astype(int).unique().tolist())
selected_years = st.sidebar.multiselect(
    "Posting year",
    options=available_years,
    default=available_years,
)

employment_options = sorted(df["employmentTypes"].dropna().astype(str).unique().tolist())
selected_employment = st.sidebar.multiselect(
    "Employment type",
    options=employment_options,
    default=employment_options,
)

position_options = sorted(df["positionLevels"].dropna().astype(str).unique().tolist())
selected_positions = st.sidebar.multiselect(
    "Position level",
    options=position_options,
    default=position_options,
)

status_options = sorted(df["status_jobStatus"].dropna().astype(str).unique().tolist())
selected_status = st.sidebar.multiselect(
    "Job status",
    options=status_options,
    default=status_options,
)

salary_cap = st.sidebar.number_input(
    "Maximum monthly salary included in benchmarking",
    min_value=1000,
    max_value=200000,
    value=30000,
    step=1000,
    help="Helps prevent extreme salary outliers from distorting benchmark averages.",
)

min_role_postings = st.sidebar.slider(
    "Minimum postings per role in role-level charts",
    min_value=1,
    max_value=100,
    value=10,
)

# Apply filters
filtered = df[
    df["posting_year"].isin(selected_years)
    & df["employmentTypes"].isin(selected_employment)
    & df["positionLevels"].isin(selected_positions)
    & df["status_jobStatus"].isin(selected_status)
].copy()

# Salary cleaning after filter
filtered.loc[
    (filtered["average_salary_clean"] <= 0)
    | (filtered["average_salary_clean"] > salary_cap),
    "average_salary_clean",
] = np.nan

if filtered.empty:
    st.warning("No data matches the current filters.")
    st.stop()


# ============================================================
# KPI row
# ============================================================
unique_postings = filtered["metadata_jobPostId"].nunique()
total_vacancies = int(filtered["numberOfVacancies"].sum())
median_salary = filtered["average_salary_clean"].median()
applications_per_vacancy = safe_divide(
    filtered["metadata_totalNumberJobApplication"].sum(),
    filtered["numberOfVacancies"].sum(),
)
view_to_application_rate = safe_divide(
    filtered["metadata_totalNumberJobApplication"].sum(),
    filtered["metadata_totalNumberOfView"].sum(),
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Job postings", f"{unique_postings:,}")
kpi2.metric("Vacancies", f"{total_vacancies:,}")
kpi3.metric("Median salary", format_currency(median_salary))
kpi4.metric("Applications / vacancy", f"{applications_per_vacancy:.2f}")
kpi5.metric("View → application rate", f"{view_to_application_rate:.1%}")


# ============================================================
# Summary tables
# ============================================================
category_summary = make_summary(filtered, "category")
role_summary = make_summary(filtered, "title")
role_summary = role_summary[role_summary["postings"] >= min_role_postings].copy()

category_summary, category_vacancy_threshold, category_app_threshold = add_hard_to_fill_flag(
    category_summary
)
role_summary, role_vacancy_threshold, role_app_threshold = add_hard_to_fill_flag(role_summary)


# ============================================================
# Main dashboard tabs
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Market Overview",
        "Salary Benchmarking",
        "Hard-to-Fill Roles",
        "Data Table",
    ]
)


with tab1:
    st.subheader("High-demand job categories and roles")

    left, right = st.columns(2)

    top_categories = category_summary.sort_values("vacancies", ascending=False).head(15)
    fig_categories = px.bar(
        top_categories,
        x="vacancies",
        y="category",
        orientation="h",
        title="Top job categories by total vacancies",
        labels={"vacancies": "Vacancies", "category": "Job category"},
    )
    fig_categories.update_layout(yaxis={"categoryorder": "total ascending"})
    left.plotly_chart(fig_categories, width='stretch')

    top_roles = role_summary.sort_values("vacancies", ascending=False).head(15)
    fig_roles = px.bar(
        top_roles,
        x="vacancies",
        y="title",
        orientation="h",
        title="Top roles by total vacancies",
        labels={"vacancies": "Vacancies", "title": "Role title"},
    )
    fig_roles.update_layout(yaxis={"categoryorder": "total ascending"})
    right.plotly_chart(fig_roles, width='stretch')

    left2, right2 = st.columns(2)

    fig_views = px.scatter(
        category_summary,
        x="total_views",
        y="total_applications",
        size="vacancies",
        hover_name="category",
        title="Category interest: views vs applications",
        labels={
            "total_views": "Total views",
            "total_applications": "Total applications",
            "vacancies": "Vacancies",
        },
    )
    left2.plotly_chart(fig_views, width='stretch')

    fig_experience = px.bar(
        category_summary.sort_values("avg_experience", ascending=False).head(15),
        x="avg_experience",
        y="category",
        orientation="h",
        title="Categories with highest average experience requirement",
        labels={"avg_experience": "Average minimum years of experience", "category": "Job category"},
    )
    fig_experience.update_layout(yaxis={"categoryorder": "total ascending"})
    right2.plotly_chart(fig_experience, width='stretch')


with tab2:
    st.subheader("Salary benchmarking")

    left, right = st.columns(2)

    salary_categories = category_summary.dropna(subset=["average_salary"]).sort_values(
        "average_salary", ascending=False
    ).head(15)
    fig_salary_categories = px.bar(
        salary_categories,
        x="average_salary",
        y="category",
        orientation="h",
        title="Top categories by average salary",
        labels={"average_salary": "Average monthly salary (S$)", "category": "Job category"},
    )
    fig_salary_categories.update_layout(yaxis={"categoryorder": "total ascending"})
    left.plotly_chart(fig_salary_categories, width='stretch')

    salary_roles = role_summary.dropna(subset=["average_salary"]).sort_values(
        "average_salary", ascending=False
    ).head(15)
    fig_salary_roles = px.bar(
        salary_roles,
        x="average_salary",
        y="title",
        orientation="h",
        title="Top roles by average salary",
        labels={"average_salary": "Average monthly salary (S$)", "title": "Role title"},
    )
    fig_salary_roles.update_layout(yaxis={"categoryorder": "total ascending"})
    right.plotly_chart(fig_salary_roles, width='stretch')

    benchmark_table = category_summary[
        [
            "category",
            "postings",
            "vacancies",
            "average_salary",
            "median_salary",
            "avg_experience",
            "applications_per_vacancy",
        ]
    ].sort_values("vacancies", ascending=False)

    st.markdown("#### Category benchmark table")
    st.dataframe(
        benchmark_table,
        width='stretch',
        hide_index=True,
        column_config={
            "average_salary": st.column_config.NumberColumn("Average salary", format="S$ %.0f"),
            "median_salary": st.column_config.NumberColumn("Median salary", format="S$ %.0f"),
            "avg_experience": st.column_config.NumberColumn("Avg. experience", format="%.1f"),
            "applications_per_vacancy": st.column_config.NumberColumn("Applications / vacancy", format="%.2f"),
        },
    )


with tab3:
    st.subheader("Recruitment prioritisation: high vacancies with low applicant response")
    st.write(
        "A role is flagged as **hard to fill** when it is in the top 25% for vacancies "
        "and bottom 25% for applications per vacancy within the current filter selection."
    )

    left, right = st.columns(2)

    fig_hard_categories = px.scatter(
        category_summary,
        x="vacancies",
        y="applications_per_vacancy",
        size="postings",
        color="hard_to_fill",
        hover_name="category",
        title="Hard-to-fill categories",
        labels={
            "vacancies": "Vacancies",
            "applications_per_vacancy": "Applications per vacancy",
            "hard_to_fill": "Hard to fill",
        },
    )
    fig_hard_categories.add_vline(x=category_vacancy_threshold, line_dash="dash")
    fig_hard_categories.add_hline(y=category_app_threshold, line_dash="dash")
    left.plotly_chart(fig_hard_categories, width='stretch')

    fig_hard_roles = px.scatter(
        role_summary,
        x="vacancies",
        y="applications_per_vacancy",
        size="postings",
        color="hard_to_fill",
        hover_name="title",
        title="Hard-to-fill roles",
        labels={
            "vacancies": "Vacancies",
            "applications_per_vacancy": "Applications per vacancy",
            "hard_to_fill": "Hard to fill",
        },
    )
    fig_hard_roles.add_vline(x=role_vacancy_threshold, line_dash="dash")
    fig_hard_roles.add_hline(y=role_app_threshold, line_dash="dash")
    right.plotly_chart(fig_hard_roles, width='stretch')

    hard_roles_table = role_summary[role_summary["hard_to_fill"]].sort_values(
        ["vacancies", "applications_per_vacancy"], ascending=[False, True]
    )[
        [
            "title",
            "postings",
            "vacancies",
            "average_salary",
            "avg_experience",
            "total_views",
            "total_applications",
            "applications_per_vacancy",
            "view_to_application_rate",
        ]
    ]

    st.markdown("#### Roles to prioritise")
    st.dataframe(
        hard_roles_table,
        width='stretch',
        hide_index=True,
        column_config={
            "average_salary": st.column_config.NumberColumn("Average salary", format="S$ %.0f"),
            "avg_experience": st.column_config.NumberColumn("Avg. experience", format="%.1f"),
            "applications_per_vacancy": st.column_config.NumberColumn("Applications / vacancy", format="%.2f"),
            "view_to_application_rate": st.column_config.NumberColumn("View → application rate", format="%.1%"),
        },
    )

    csv = hard_roles_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download prioritised roles as CSV",
        data=csv,
        file_name="prioritised_roles.csv",
        mime="text/csv",
    )


with tab4:
    st.subheader("Filtered posting-level data")
    display_cols = [
        "posting_date",
        "category",
        "title",
        "employmentTypes",
        "positionLevels",
        "numberOfVacancies",
        "average_salary_clean",
        "minimumYearsExperience",
        "metadata_totalNumberOfView",
        "metadata_totalNumberJobApplication",
        "postedCompany_name",
    ]

    st.dataframe(
        filtered[display_cols].sort_values("posting_date", ascending=False),
        width='stretch',
        hide_index=True,
        column_config={
            "posting_date": st.column_config.DateColumn("Posting date"),
            "average_salary_clean": st.column_config.NumberColumn("Average salary", format="S$ %.0f"),
            "minimumYearsExperience": st.column_config.NumberColumn("Min. years experience", format="%.0f"),
            "metadata_totalNumberOfView": st.column_config.NumberColumn("Views"),
            "metadata_totalNumberJobApplication": st.column_config.NumberColumn("Applications"),
        },
    )


# ============================================================
# Footer note
# ============================================================
st.divider()
st.caption(
    "Suggested interpretation: prioritise roles with many vacancies but low applications per vacancy; "
    "review whether salary benchmarks, experience requirements, or sourcing channels may need adjustment."
)
