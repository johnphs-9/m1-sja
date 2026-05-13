import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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

SQL_QUERY = """
    SELECT
        j.listing_id,
        j.metadata_jobPostId,
        j.title,
        j.employmentTypes,
        j.positionLevels,
        j.postedCompany_name,
        j.metadata_newPostingDate,
        j.metadata_originalPostingDate,
        j.metadata_expiryDate,
        j.metadata_repostCount,
        j.minimumYearsExperience,
        j.numberOfVacancies,
        j.salary_minimum,
        j.salary_maximum,
        j.salary_type,
        j.average_salary,
        j.metadata_totalNumberOfView,
        j.metadata_totalNumberJobApplication,
        j.status_jobStatus,
        c.category_id,
        c.category_name
    FROM sg_job_data AS j
    INNER JOIN job_listing_categories AS jc
        ON j.listing_id = jc.listing_id
    INNER JOIN categories AS c
        ON jc.category_id = c.category_id
"""

@st.cache_data(show_spinner=True)
def load_data(db_path: str) -> pd.DataFrame:
    """
    Load data from a normalised DuckDB schema:
    - sg_job_data
    - categories
    - job_listing_categories (bridging table for many-to-many relationship)
  

    The returned dataframe has one row per job-category relationship.
    This means a job assigned to 3 categories appears in 3 rows.
    """

    with duckdb.connect(db_path, read_only=True) as con:
        df = con.execute(SQL_QUERY).df()

    return prepare_data(df)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["posting_date"] = pd.to_datetime(
        data["metadata_newPostingDate"], errors="coerce"
    )
    data["posting_year"] = data["posting_date"].dt.year

    text_cols = [
        "title",
        "employmentTypes",
        "positionLevels",
        "postedCompany_name",
        "status_jobStatus",
        "category_name",
    ]
    for col in text_cols:
        if col in data.columns:
            data[col] = data[col].fillna("Missing").astype(str)

    numeric_cols = [
        "metadata_repostCount",
        "minimumYearsExperience",
        "numberOfVacancies",
        "salary_minimum",
        "salary_maximum",
        "average_salary",
        "metadata_totalNumberOfView",
        "metadata_totalNumberJobApplication",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["average_salary_clean"] = data["average_salary"].where(
        data["average_salary"] > 0
    )

    data["applications_per_vacancy"] = np.where(
        data["numberOfVacancies"] > 0,
        data["metadata_totalNumberJobApplication"] / data["numberOfVacancies"],
        np.nan,
    )

    data["view_to_application_rate"] = np.where(
        data["metadata_totalNumberOfView"] > 0,
        data["metadata_totalNumberJobApplication"] / data["metadata_totalNumberOfView"] * 100,
        np.nan,
    )

    return data


def format_currency(value):
    if pd.isna(value):
        return "N/A"
    return f"S${value:,.0f}"


def safe_divide(numerator, denominator):
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def make_summary(data: pd.DataFrame, group_col: str) -> pd.DataFrame:
    summary = (
        data.groupby(group_col, dropna=False)
        .agg(
            job_postings=("metadata_jobPostId", "nunique"),
            vacancies=("numberOfVacancies", "sum"),
            average_salary=("average_salary_clean", "mean"),
            median_salary=("average_salary_clean", "median"),
            avg_min_experience=("minimumYearsExperience", "mean"),
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
        summary["total_applications"] / summary["total_views"] * 100,
        np.nan,
    )

    return summary


def add_hard_to_fill_flag(
    summary: pd.DataFrame,
    vacancy_quantile: float = 0.75,
    application_quantile: float = 0.25,
):
    result = summary.copy()

    if result.empty:
        result["hard_to_fill"] = False
        return result, np.nan, np.nan

    vacancy_threshold = result["vacancies"].quantile(vacancy_quantile)
    app_threshold = result["applications_per_vacancy"].quantile(application_quantile)

    result["hard_to_fill"] = (
        (result["vacancies"] >= vacancy_threshold)
        & (result["applications_per_vacancy"] <= app_threshold)
    )

    return result, vacancy_threshold, app_threshold


st.sidebar.header("DuckDB source")

db_path = st.sidebar.text_input(
    "DuckDB database path",
    #value="SGJobData.duckdb",
    value="sgJobData-normalised.db",
    help="Example: SGJobData.db or db/jobs.duckdb",
)

try:
    df = load_data(db_path)
except Exception as e:
    st.error(
        "Could not load the DuckDB database. Check the file path and confirm that "
        "the tables sg_job_data, job_listing_categories, and categories exist."
    )
    st.exception(e)
    st.stop()

st.sidebar.success(f"Loaded {df['metadata_jobPostId'].nunique():,} unique job postings")

st.sidebar.header("Filters")

available_years = sorted(df["posting_year"].dropna().astype(int).unique().tolist())
selected_years = st.sidebar.multiselect(
    "Posting year",
    options=available_years,
    default=available_years,
)

category_options = sorted(df["category_name"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Job category",
    options=category_options,
    default=category_options[:10]  # Default to top 5 categories for better performance on initial load
)
#print(f"Selected categories: {selected_categories}")  # Debug print to check selected categories

employment_options = sorted(df["employmentTypes"].dropna().unique().tolist())
selected_employment = st.sidebar.multiselect(
    "Employment type",
    options=employment_options,
    default=employment_options,
)

position_options = sorted(df["positionLevels"].dropna().unique().tolist())
selected_positions = st.sidebar.multiselect(
    "Position level",
    options=position_options,
    default=position_options,
)

status_options = sorted(df["status_jobStatus"].dropna().unique().tolist())
selected_status = st.sidebar.multiselect(
    "Job status",
    options=status_options,
    default=status_options[1:],
)

salary_cap = st.sidebar.number_input(
    "Maximum monthly salary included in benchmarking",
    min_value=1000,
    max_value=200000,
    value=30000,
    step=1000,
    help="Helps prevent extreme salary values from distorting salary benchmarks.",
)

min_role_postings = st.sidebar.slider(
    "Minimum postings per role in role charts",
    min_value=1,
    max_value=100,
    value=10,
    help="Filters out roles with fewer than the specified number of postings with duplicate postings excluded."
)

filtered = df[
    df["posting_year"].isin(selected_years)
    & df["category_name"].isin(selected_categories)
    & df["employmentTypes"].isin(selected_employment)
    & df["positionLevels"].isin(selected_positions)
    & df["status_jobStatus"].isin(selected_status)
].copy()

# Set extreme salary values to NaN to avoid distortion in benchmarks and charts.
filtered.loc[
    (filtered["average_salary_clean"] <= 0)
    | (filtered["average_salary_clean"] > salary_cap),
    "average_salary_clean",
] = np.nan

if filtered.empty:
    st.warning("No records match the current filters.")
    st.stop()

# Overall KPIs use unique job postings to avoid double counting multi-category jobs.
unique_jobs = filtered["metadata_jobPostId"].nunique()
unique_categories = filtered["category_name"].nunique()
dedup_jobs = filtered.drop_duplicates("metadata_jobPostId")
total_vacancies = dedup_jobs["numberOfVacancies"].sum()
median_salary = dedup_jobs["average_salary_clean"].median()
applications_per_vacancy = safe_divide(
    dedup_jobs["metadata_totalNumberJobApplication"].sum(),
    dedup_jobs["numberOfVacancies"].sum(),
)
view_to_application_rate = safe_divide(
    dedup_jobs["metadata_totalNumberJobApplication"].sum(),
    dedup_jobs["metadata_totalNumberOfView"].sum(),
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Unique job postings", f"{unique_jobs:,}")
kpi2.metric("Categories shown", f"{unique_categories:,}")
kpi3.metric("Total vacancies", f"{int(total_vacancies):,}", help="Total number of vacancies across all unique job postings. Some postings may have multiple vacancies, but duplicated postings are removed.")
kpi4.metric("Median salary", format_currency(median_salary))
kpi5.metric("Applications / vacancy", f"{applications_per_vacancy:.1%}", help="Lower values indicate fewer applications per vacancy, suggesting harder-to-fill roles.")

st.caption(
    "Note: Category-level charts count a multi-category posting once per assigned category. "
    "Overall KPIs use unique job postings to avoid double counting."
)

category_summary = make_summary(filtered, "category_name")
role_summary = make_summary(filtered, "title")
role_summary = role_summary[role_summary["job_postings"] >= min_role_postings].copy()

category_summary, cat_vacancy_threshold, cat_app_threshold = add_hard_to_fill_flag(
    category_summary
)
role_summary, role_vacancy_threshold, role_app_threshold = add_hard_to_fill_flag(
    role_summary
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Market Overview",
        "Salary Benchmarking",
        "Hard-to-Fill Roles",
        "Recruitment Table",
        "SQL Reference",
    ]
)

with tab1:
    st.subheader("High-demand job categories and roles")

    left, right = st.columns(2)

    top_categories = category_summary.sort_values("vacancies", ascending=False).head(15)
    fig_categories = px.bar(
        top_categories,
        x="vacancies",
        y="category_name",
        orientation="h",
        title="Top job categories by total vacancies",
        labels={"vacancies": "Vacancies", "category_name": "Job category"},
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

    fig_interest = px.scatter(
        category_summary,
        x="total_views",
        y="total_applications",
        size="vacancies",
        hover_name="category_name",
        title="Category interest: views vs applications",
        labels={
            "total_views": "Total views",
            "total_applications": "Total applications",
            "vacancies": "Vacancies",
        },
    )
    left2.plotly_chart(fig_interest, width='stretch')

    fig_exp = px.bar(
        category_summary.sort_values("avg_min_experience", ascending=False).head(15),
        x="avg_min_experience",
        y="category_name",
        orientation="h",
        title="Categories with highest average experience requirement",
        labels={
            "avg_min_experience": "Average minimum years of experience",
            "category_name": "Job category",
        },
    )
    fig_exp.update_layout(yaxis={"categoryorder": "total ascending"})
    right2.plotly_chart(fig_exp, width='stretch')

with tab2:
    st.subheader("Salary benchmarking")

    left, right = st.columns(2)

    salary_categories = category_summary.dropna(subset=["average_salary"]).sort_values(
        "average_salary", ascending=False
    ).head(15)
    fig_salary_categories = px.bar(
        salary_categories,
        x="average_salary",
        y="category_name",
        orientation="h",
        title="Top categories by average salary",
        labels={
            "average_salary": "Average monthly salary (S$)",
            "category_name": "Job category",
        },
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

    st.markdown("#### Category salary benchmark table")
    benchmark_table = category_summary[
        [
            "category_name",
            "job_postings",
            "vacancies",
            "average_salary",
            "median_salary",
            "avg_min_experience",
            "applications_per_vacancy",
            "view_to_application_rate",
        ]
    ].sort_values("vacancies", ascending=False)

    st.dataframe(
        benchmark_table,
        width='stretch',
        hide_index=True,
        column_config={
            "category_name": "Category",
            "average_salary": st.column_config.NumberColumn("Average salary", format="S$ %.0f"),
            "median_salary": st.column_config.NumberColumn("Median salary", format="S$ %.0f"),
            "avg_min_experience": st.column_config.NumberColumn("Avg. min experience", format="%.1f"),
            "applications_per_vacancy": st.column_config.NumberColumn("Applications / vacancy", format="%.3f", help="Lower values indicate fewer applications per vacancy, suggesting harder-to-fill categories."),
            "view_to_application_rate": st.column_config.NumberColumn("View → application rate", format="%.1f%%", help="Lower values indicate fewer applications per view, suggesting lower applicant interest relative to visibility."),
        },
    )

with tab3:
    st.subheader("Recruitment prioritisation: high vacancies with low applicant response")
    st.write(
        "A role or category is flagged as **hard to fill** when it is in the top 25% "
        "for vacancies and bottom 25% for applications per vacancy within the current filters."
    )

    left, right = st.columns(2)

    fig_hard_cat = px.scatter(
        category_summary,
        x="vacancies",
        y="applications_per_vacancy",
        size="job_postings",
        color="hard_to_fill",
        hover_name="category_name",
        title="Hard-to-fill categories",
        labels={
            "vacancies": "Vacancies",
            "applications_per_vacancy": "Applications per vacancy",
            "hard_to_fill": "Hard to fill",
        },
    )
    fig_hard_cat.add_vline(x=cat_vacancy_threshold, line_dash="dash")
    fig_hard_cat.add_hline(y=cat_app_threshold, line_dash="dash")
    left.plotly_chart(fig_hard_cat, width='stretch')

    fig_hard_role = px.scatter(
        role_summary,
        x="vacancies",
        y="applications_per_vacancy",
        size="job_postings",
        color="hard_to_fill",
        hover_name="title",
        title="Hard-to-fill roles",
        labels={
            "vacancies": "Vacancies",
            "applications_per_vacancy": "Applications per vacancy",
            "hard_to_fill": "Hard to fill",
        },
    )
    fig_hard_role.add_vline(x=role_vacancy_threshold, line_dash="dash")
    fig_hard_role.add_hline(y=role_app_threshold, line_dash="dash")
    right.plotly_chart(fig_hard_role, width='stretch')

    hard_roles_table = role_summary[role_summary["hard_to_fill"]].sort_values(
        ["vacancies", "applications_per_vacancy"], ascending=[False, True]
    )[
        [
            "title",
            "job_postings",
            "vacancies",
            "average_salary",
            "avg_min_experience",
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
            "avg_min_experience": st.column_config.NumberColumn("Avg. min experience", format="%.1f"),
            "applications_per_vacancy": st.column_config.NumberColumn("Applications / vacancy", format="%.3f", help="Lower values indicate fewer applications per vacancy, suggesting harder-to-fill roles."),
            "view_to_application_rate": st.column_config.NumberColumn("View → application rate", format="%.1f%%", help="Lower values indicate fewer applications per view, suggesting lower applicant interest relative to visibility."),
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
    st.subheader("Filtered recruitment data")

    display_cols = [
        "posting_date",
        "category_name",
        "title",
        "employmentTypes",
        "positionLevels",
        "numberOfVacancies",
        "average_salary_clean",
        "minimumYearsExperience",
        "metadata_totalNumberOfView",
        "metadata_totalNumberJobApplication",
        "postedCompany_name",
        "status_jobStatus",
    ]

    display_data = filtered[display_cols].sort_values("posting_date", ascending=False)

    st.dataframe(
        display_data,
        width='stretch',
        hide_index=True,
        column_config={
            "posting_date": st.column_config.DateColumn("Posting date"),
            "category_name": "Category",
            "numberOfVacancies": st.column_config.NumberColumn("Vacancies"),
            "average_salary_clean": st.column_config.NumberColumn("Average salary", format="S$ %.0f"),
            "minimumYearsExperience": st.column_config.NumberColumn("Min. years experience", format="%.0f"),
            "metadata_totalNumberOfView": st.column_config.NumberColumn("Views"),
            "metadata_totalNumberJobApplication": st.column_config.NumberColumn("Applications"),
        },
    )

    csv = display_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv,
        file_name="filtered_recruitment_data.csv",
        mime="text/csv",
    )

with tab5:
    st.subheader("DuckDB SQL used by the dashboard")
    st.write(
        "The app reads from the normalised tables using this join. "
        "This avoids parsing the original JSON-like categories column inside Streamlit."
    )

    st.code(SQL_QUERY, language="sql")



st.divider()
st.caption(
    "Suggested interpretation: prioritise roles with many vacancies but low applications per vacancy; "
    "then review salary benchmarks, experience requirements, and sourcing channels."
)
