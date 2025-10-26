import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ------------------------
# Load model and pipeline
# ------------------------
with open("model/best_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/full_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# ------------------------
# App title
# ------------------------
st.title("💡 Health Insurance Cost Predictor")

st.sidebar.header("User Input Features")

# ------------------------
# User input
# ------------------------
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.sidebar.number_input(
    "Number of Children", min_value=0, max_value=10, value=0
)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
smoker = st.sidebar.selectbox("Smoker?", ["no", "yes"])
region = st.sidebar.selectbox(
    "Region", ["southwest", "southeast", "northwest", "northeast"]
)

new_data = pd.DataFrame(
    [
        {
            "age": age,
            "bmi": bmi,
            "children": children,
            "sex": sex,
            "smoker": smoker,
            "region": region,
        }
    ]
)

# ------------------------
# Predict
# ------------------------
new_data_prepared = pipeline.transform(new_data)
predicted_annual = model.predict(new_data_prepared)[0]
predicted_monthly = predicted_annual / 12

st.subheader("💰 Estimated Insurance Cost")
st.write(f"Annual: ${predicted_annual:,.2f}")
st.write(f"Monthly: ${predicted_monthly:,.2f}")

# ------------------------
# Feature Importance
# ------------------------
st.subheader("📊 Feature Importances")

num_attribs = ["age", "bmi", "children"]
cat_attribs = ["sex", "smoker", "region"]

cat_encoder = pipeline.named_transformers_["cat"].named_steps["encoder"]
cat_one_hot_attribs = cat_encoder.get_feature_names_out(cat_attribs)
all_features = num_attribs + list(cat_one_hot_attribs)

feat_importances = pd.Series(model.feature_importances_, index=all_features)
feat_importances = feat_importances.sort_values(ascending=True)

st.bar_chart(feat_importances)

# ------------------------
# Scatter plots: numeric features vs charges
# ------------------------
st.subheader("📈 Numeric Features vs Charges")

df_plot = pd.read_csv("data/insurance.csv")
numeric_features = ["age", "bmi", "children"]

plt.figure(figsize=(24, 8))
for i, feature in enumerate(numeric_features):
    plt.subplot(1, 3, i + 1)

    smokers_df = df_plot[df_plot["smoker"] == "yes"]
    non_smokers_df = df_plot[df_plot["smoker"] == "no"]

    plt.scatter(
        non_smokers_df[feature],
        non_smokers_df["charges"],
        c="skyblue",
        alpha=0.6,
        s=50,
        label="Non-Smoker",
        edgecolors="w",
    )
    plt.scatter(
        smokers_df[feature],
        smokers_df["charges"],
        c="salmon",
        alpha=0.6,
        s=50,
        label="Smoker",
        edgecolors="w",
    )
    # User input point
    plt.scatter(
        new_data[feature],
        predicted_annual,
        c="green",
        s=150,
        label="Your Input",
        edgecolor="black",
        marker="*",
    )

    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Charges", fontsize=12)
    plt.title(f"{feature} vs Charges", fontsize=14)
    if i == 0:
        plt.legend(fontsize=12)

plt.tight_layout()
st.pyplot(plt)
