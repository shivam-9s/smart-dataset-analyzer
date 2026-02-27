import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def show_basic_info(X, y):
    st.subheader("📊 Dataset Overview")
    st.write("Shape:", X.shape)
    st.write("Columns:", list(X.columns))
    st.write("Missing Values:")
    st.write(X.isnull().sum())
    st.write("Statistical Summary:")
    st.write(X.describe())

def show_target_distribution(y):
    st.subheader("🎯 Target Distribution")
    st.write(y.value_counts())

def show_feature_distribution(X):
    st.subheader("📈 Feature Visualization")
    feature = st.selectbox("Select Feature", X.columns)

    fig, ax = plt.subplots()
    sns.histplot(X[feature], kde=True, ax=ax)
    st.pyplot(fig)

def show_correlation(X):
    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(X.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
