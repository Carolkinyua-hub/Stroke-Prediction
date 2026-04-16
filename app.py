# app.py
# Streamlit version of your working notebook logic
# Upload CSV -> remove IDs -> FAMD(8) -> KMeans(4) -> reattach IDs -> profiles + visuals

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import prince

from sklearn.cluster import KMeans

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="SME Clustering App",
    layout="wide"
)

st.title("SME Clustering App")
st.caption(
    "Upload a CSV file to cluster SMEs using FAMD + KMeans."
)

# =====================================================
# FIXED SETTINGS
# =====================================================
N_COMPONENTS = 8
N_CLUSTERS = 4

id_cols = [
    "Company name",
    "Company Email",
    "Company number",
    "City"
]

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# =====================================================
# MAIN APP
# =====================================================
if uploaded_file is not None:

    try:
        # ---------------------------------------------
        # LOAD DATA
        # ---------------------------------------------
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

        # ---------------------------------------------
        # IDENTIFY ID COLUMNS PRESENT
        # ---------------------------------------------
        existing_ids = [col for col in id_cols if col in df.columns]

        ids = df[existing_ids].copy()

        # ---------------------------------------------
        # REMOVE IDS
        # ---------------------------------------------
        X = df.drop(columns=existing_ids, errors="ignore").copy()

        # ---------------------------------------------
        # RUN MODEL
        # ---------------------------------------------
        if st.button("Run Analysis"):

            with st.spinner("Running clustering model..."):

                # FAMD
                famd = prince.FAMD(
                    n_components=N_COMPONENTS,
                    random_state=42
                )

                X_famd = famd.fit_transform(X)

                # KMeans
                kmeans = KMeans(
                    n_clusters=N_CLUSTERS,
                    random_state=42,
                    n_init=10
                )

                clusters = kmeans.fit_predict(X_famd)

            # -----------------------------------------
            # FAMD COORDINATES
            # -----------------------------------------
            X_famd_df = X_famd.copy()
            X_famd_df.columns = [
                f"dim_{i}" for i in range(X_famd_df.shape[1])
            ]

            # -----------------------------------------
            # CLUSTER LABELS
            # -----------------------------------------
            clusters_df = pd.DataFrame(
                {"cluster": clusters},
                index=X.index
            )

            # -----------------------------------------
            # REATTACH IDS
            # -----------------------------------------
            result_df = pd.concat(
                [
                    ids.reset_index(drop=True),
                    X_famd_df.reset_index(drop=True),
                    clusters_df.reset_index(drop=True)
                ],
                axis=1
            )

            # -----------------------------------------
            # SHOW RESULTS
            # -----------------------------------------
            st.subheader("Clustered Results")
            st.dataframe(result_df)

            # -----------------------------------------
            # NUMERIC PROFILE
            # -----------------------------------------
            numerical_cols = X.select_dtypes(
                include=["number"]
            ).columns

            if len(numerical_cols) > 0:

                cluster_profile_num = X.groupby(
                    clusters
                )[numerical_cols].mean()

                st.subheader("Numeric Cluster Profile")
                st.dataframe(cluster_profile_num)

            # -----------------------------------------
            # CATEGORICAL PROFILE
            # -----------------------------------------
            categorical_cols = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            if len(categorical_cols) > 0:

                st.subheader("Categorical Visualisation")

                selected_col = st.selectbox(
                    "Choose categorical feature",
                    categorical_cols
                )

                cross = pd.crosstab(
                    clusters,
                    X[selected_col],
                    normalize="index"
                )

                st.dataframe(cross)

                fig, ax = plt.subplots(
                    figsize=(12, 6)
                )

                cross.plot(
                    kind="bar",
                    stacked=True,
                    ax=ax
                )

                ax.set_title(
                    f"{selected_col} by Cluster"
                )

                ax.set_xlabel("Cluster")
                ax.set_ylabel("Proportion")

                plt.xticks(rotation=0)
                plt.tight_layout()

                st.pyplot(fig)

            # -----------------------------------------
            # CLUSTER COUNTS
            # -----------------------------------------
            st.subheader("Cluster Counts")

            counts = result_df["cluster"] \
                .value_counts() \
                .sort_index()

            st.bar_chart(counts)

            # -----------------------------------------
            # FAMD SCATTER
            # -----------------------------------------
            st.subheader("Cluster Map")

            fig2, ax2 = plt.subplots(
                figsize=(8, 5)
            )

            ax2.scatter(
                X_famd_df["dim_0"],
                X_famd_df["dim_1"],
                c=clusters
            )

            ax2.set_xlabel("Dimension 1")
            ax2.set_ylabel("Dimension 2")
            ax2.set_title("FAMD Cluster Projection")

            st.pyplot(fig2)

            # -----------------------------------------
            # VIEW CLUSTER MEMBERS
            # -----------------------------------------
            st.subheader("View SMEs by Cluster")

            selected_cluster = st.selectbox(
                "Choose cluster",
                sorted(result_df["cluster"].unique())
            )

            st.dataframe(
                result_df[
                    result_df["cluster"] == selected_cluster
                ]
            )

            # -----------------------------------------
            # DOWNLOAD
            # -----------------------------------------
            csv = result_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name="cluster_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(str(e))

else:
    st.info("Upload a CSV file to begin.")
