import streamlit as st
from dataset_loader import load_dataset
from eda import (
    show_basic_info,
    show_target_distribution,
    show_feature_distribution,
    show_correlation
)
from utils import detect_problem_type, analyze_dataset
from model_training import train_model
from evaluation import evaluate_model

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Smart Dataset Analyzer", layout="wide")

st.title("🚀 Smart Dataset Analyzer & ML Recommender")

# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------
if "trained" not in st.session_state:
    st.session_state.trained = False

if "model_results" not in st.session_state:
    st.session_state.model_results = {}

if "user_note" not in st.session_state:
    st.session_state.user_note = ""

# --------------------------------------------------
# Dataset Selection (Improved)
# --------------------------------------------------

dataset_options = [
    "-- Select Dataset --",
    "Iris",
    "Wine",
    "Breast Cancer",
    "Diabetes",
    "California Housing"
]

dataset_name = st.selectbox(
    "Select Dataset",
    dataset_options,
    index=0
)

# If no dataset selected → stop execution
if dataset_name == "-- Select Dataset --":
    st.info("👈 Please select a dataset to begin analysis.")
    st.stop()

# Reset model state if dataset changes
if "previous_dataset" not in st.session_state:
    st.session_state.previous_dataset = dataset_name

if dataset_name != st.session_state.previous_dataset:
    st.session_state.trained = False
    st.session_state.model_results = {}
    st.session_state.user_note = ""
    st.session_state.previous_dataset = dataset_name

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
X, y = load_dataset(dataset_name)

if X is not None:

    # --------------------------------------------------
    # 📊 EDA SECTION
    # --------------------------------------------------
    show_basic_info(X, y)
    show_target_distribution(y)
    show_feature_distribution(X)
    show_correlation(X)

    # --------------------------------------------------
    # 🧠 Problem Detection
    # --------------------------------------------------
    problem_type = detect_problem_type(y)

    st.subheader("🧠 Problem Type Detected")
    st.success(problem_type)

    # --------------------------------------------------
    # 🤖 Data-Driven Recommendation
    # --------------------------------------------------
    recommended_model, analysis_reason = analyze_dataset(X, y, problem_type)

    st.subheader("🤖 Data-Driven Model Recommendation")
    st.success(recommended_model)

    st.subheader("📌 Reason Based on Dataset Analysis")
    st.info(analysis_reason)

    # --------------------------------------------------
    # 📋 Model Selection
    # --------------------------------------------------
    if problem_type == "Classification":
        model_list = ["Logistic Regression", "Random Forest", "SVM"]
    else:
        model_list = ["Linear Regression", "Random Forest"]

    default_index = model_list.index(recommended_model) if recommended_model in model_list else 0

    model_name = st.selectbox(
        "Choose Model",
        model_list,
        index=default_index
    )

    # --------------------------------------------------
    # 🚀 Train Model
    # --------------------------------------------------
    if st.button("Train Model", key="train_button"):

        model, X_test, y_test, predictions = train_model(
            X, y, problem_type, model_name
        )

        score = evaluate_model(problem_type, y_test, predictions)

        st.session_state.trained = True
        st.session_state.model_results = {
            "problem_type": problem_type,
            "score": score
        }

    # --------------------------------------------------
    # 📊 Display Results
    # --------------------------------------------------
    if st.session_state.trained:

        saved_problem_type = st.session_state.model_results["problem_type"]
        saved_score = st.session_state.model_results["score"]

        st.subheader("📌 Final Conclusion")

        if saved_problem_type == "Classification":
            if saved_score > 0.9:
                st.success("Excellent model performance.")
            elif saved_score > 0.7:
                st.info("Good model performance.")
            else:
                st.warning("Model performance is limited.")
        else:
            if saved_score > 0.8:
                st.success("Strong predictive performance.")
            elif saved_score > 0.5:
                st.info("Moderate predictive performance.")
            else:
                st.warning("Low predictive performance.")

        # --------------------------------------------------
        # 📝 User Explanation
        # --------------------------------------------------
        st.subheader("📝 Add Your Own Explanation")

        user_explanation = st.text_area(
            "Write your interpretation of the model results here:",
            value=st.session_state.user_note,
            height=150,
            key="explanation_box"
        )

        if st.button("Submit Explanation", key="submit_explanation"):
            if user_explanation.strip() == "":
                st.warning("Please write something before submitting.")
            else:
                st.session_state.user_note = user_explanation
                st.success("Your explanation has been saved successfully!")

        if st.session_state.user_note != "":
            st.markdown("### 📌 Your Saved Explanation:")
            st.info(st.session_state.user_note)
