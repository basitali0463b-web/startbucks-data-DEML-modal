
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="Starbucks EDA",
    page_icon="☕",
    layout="wide"
)

st.title("☕ Starbucks Dataset - Exploratory Data Analysis")

# Load CSV from the app's folder
file_path = Path(__file__).parent / "starbucks.csv"

try:
    df = pd.read_csv(file_path)
    st.success("Dataset loaded successfully!")
except FileNotFoundError:
    st.error(
        "starbucks.csv not found. Make sure it is uploaded "
        "to the same GitHub repository folder as app.py."
    )
    st.stop()

# --- Dataset Preview ---
st.header("First 5 Rows")
st.dataframe(df.head())

# --- Dataset Information ---
st.header("Dataset Information")
col1, col2 = st.columns(2)

with col1:
    st.metric("Total Rows", df.shape[0])

with col2:
    st.metric("Total Columns", df.shape[1])

st.subheader("Column Data Types")
info_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing Values": df.isnull().sum().values
})
st.dataframe(info_df, use_container_width=True)

# --- Descriptive Statistics ---
st.header("Descriptive Statistics")
st.dataframe(df.describe(include="all").fillna(""))

# --- Pie Chart ---
st.header("Beverage Category Distribution")

if "Beverage_category" in df.columns:
    category_counts = df["Beverage_category"].value_counts()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.pie(
        category_counts,
        labels=category_counts.index,
        autopct="%1.1f%%",
        startangle=140
    )
    ax.set_title("Distribution of Starbucks Beverage Categories")
    ax.axis("equal")

    st.pyplot(fig)
    plt.close(fig)

else:
    st.warning(
        "The 'Beverage_category' column was not found in the CSV."
    )

# --- Numeric Correlation Heatmap ---
st.header("Numeric Feature Correlation")

numeric_df = df.select_dtypes(include="number")

if numeric_df.shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Starbucks Nutrition Feature Correlation")
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Not enough numeric columns to create a correlation heatmap.")
