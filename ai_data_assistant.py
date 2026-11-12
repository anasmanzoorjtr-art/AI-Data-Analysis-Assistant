
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Data Analysis Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("AI Data Analysis Assistant")
st.write("Upload a CSV file and ask questions about your data.")

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.success("Dataset loaded successfully!")

        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", len(df))
        col2.metric("Total Columns", len(df.columns))
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.write("### Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        st.write("### Ask a Question")

        question = st.text_input(
            "Enter your question",
            placeholder="Example: What is the average salary?"
        )

        if st.button("Analyze Question"):
            q = question.lower().strip()

            if not q:
                st.warning("Please enter a question.")

            elif any(word in q for word in ["summary", "overview", "describe"]):
                st.write("### Dataset Summary")
                st.dataframe(df.describe(include="all").T)

            elif "missing" in q or "null" in q:
                st.write("### Missing Values")
                missing = df.isnull().sum()
                st.dataframe(
                    missing[missing > 0].rename("Missing Count")
                )
                if missing.sum() == 0:
                    st.success("No missing values found.")

            elif "duplicate" in q:
                st.write("Duplicate rows:", int(df.duplicated().sum()))

            elif "average" in q or "mean" in q:
                if numeric_cols:
                    selected = next(
                        (c for c in numeric_cols if c.lower() in q),
                        numeric_cols[0]
                    )
                    st.write(f"Average of {selected}:")
                    st.metric("Mean", round(df[selected].mean(), 2))
                else:
                    st.warning("No numeric columns found.")

            elif "highest" in q or "maximum" in q or "max" in q:
                if numeric_cols:
                    selected = next(
                        (c for c in numeric_cols if c.lower() in q),
                        numeric_cols[0]
                    )
                    idx = df[selected].idxmax()
                    st.write(f"Highest value in {selected}:")
                    st.dataframe(df.loc[[idx]])
                else:
                    st.warning("No numeric columns found.")

            elif "lowest" in q or "minimum" in q or "min" in q:
                if numeric_cols:
                    selected = next(
                        (c for c in numeric_cols if c.lower() in q),
                        numeric_cols[0]
                    )
                    idx = df[selected].idxmin()
                    st.write(f"Lowest value in {selected}:")
                    st.dataframe(df.loc[[idx]])
                else:
                    st.warning("No numeric columns found.")

            elif "correlation" in q or "relationship" in q:
                if len(numeric_cols) >= 2:
                    st.write("### Correlation Analysis")
                    st.dataframe(df[numeric_cols].corr().round(2))
                else:
                    st.warning("At least two numeric columns are required.")

            elif "count" in q or "frequency" in q or "distribution" in q:
                if categorical_cols:
                    selected = next(
                        (c for c in categorical_cols if c.lower() in q),
                        categorical_cols[0]
                    )
                    st.write(f"Frequency of {selected}:")
                    st.dataframe(
                        df[selected].value_counts().reset_index()
                    )
                else:
                    st.warning("No categorical columns found.")

            else:
                st.info(
                    "Question not recognized. Try: summary, missing values, "
                    "average, highest, lowest, correlation, or frequency."
                )

        st.write("### Data Visualization")

        if numeric_cols:
            chart_column = st.selectbox(
                "Select a numeric column",
                numeric_cols
            )

            chart_type = st.selectbox(
                "Select chart type",
                ["Histogram", "Box Plot"]
            )

            if st.button("Generate Chart"):
                fig, ax = plt.subplots(figsize=(9, 4))

                if chart_type == "Histogram":
                    ax.hist(
                        df[chart_column].dropna(),
                        bins=20,
                        edgecolor="black"
                    )
                    ax.set_ylabel("Frequency")

                else:
                    ax.boxplot(df[chart_column].dropna())
                    ax.set_ylabel(chart_column)

                ax.set_title(f"{chart_type} of {chart_column}")
                ax.set_xlabel(chart_column)
                st.pyplot(fig)
                plt.close(fig)

        else:
            st.warning("No numeric columns available for visualization.")

        st.write("### Automated Findings")

        if numeric_cols:
            best_column = df[numeric_cols].mean().idxmax()
            st.write(
                f"- Highest numerical column average: **{best_column}**"
            )

        if categorical_cols:
            common_column = categorical_cols[0]
            common_value = df[common_column].mode()

            if not common_value.empty:
                st.write(
                    f"- Most frequent {common_column}: **{common_value.iloc[0]}**"
                )

        st.write(
            "Note: This assistant uses predefined question patterns "
            "and statistical analysis. It is not a generative AI language model."
        )

    except Exception as error:
        st.error(f"Unable to analyze the file: {error}")

else:
    st.info("Upload a CSV file to begin data analysis.")
