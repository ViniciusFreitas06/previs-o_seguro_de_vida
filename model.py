# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# ------------------------
# Load data
# ------------------------
df = pd.read_csv("data/insurance.csv")
print(df.head())
print(df.info())
print(df.describe())
print("Null values per column:\n", df.isnull().sum())
print("Smoker distribution:\n", df["smoker"].value_counts())

# ------------------------
# Histograms of numeric features
# ------------------------
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.hist(df["age"], bins=20, color="skyblue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.subplot(1, 3, 2)
plt.hist(df["bmi"], bins=20, color="lightgreen")
plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Count")

plt.subplot(1, 3, 3)
plt.hist(df["children"], bins=6, color="salmon")
plt.title("Number of Children")
plt.xlabel("Children")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# %%
# Copy dataframe and encode categorical features
df_num = df.copy()
df_num["sex"] = df_num["sex"].map({"male": 0, "female": 1})
df_num["smoker"] = df_num["smoker"].map({"no": 0, "yes": 1})
df_num["region"] = df_num["region"].map(
    {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}
)

# Correlation matrix
print("Correlation matrix:\n", df_num.corr())

# %%
# Split data into training and testing
from sklearn.model_selection import train_test_split

X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
# Define numeric and categorical columns
numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Numeric pipeline
numeric_pipeline = Pipeline([("scaler", StandardScaler())])

# Categorical pipeline
categorical_pipeline = Pipeline([("encoder", OneHotEncoder(drop="first"))])

# Full pipeline
full_pipeline = ColumnTransformer(
    [
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)

X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)

# %%
# Linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

linear_model = LinearRegression()
linear_model.fit(X_train_prepared, y_train)

y_pred_linear = linear_model.predict(X_test_prepared)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
print(f"Linear Regression RMSE: {rmse_linear:.2f}")

# %%
# Random Forest model
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=100, random_state=42)
forest.fit(X_train_prepared, y_train)

y_pred_forest = forest.predict(X_test_prepared)
rmse_forest = np.sqrt(mean_squared_error(y_test, y_pred_forest))
print(f"Random Forest RMSE: {rmse_forest:.2f}")

# %%
# Feature importance
cat_encoder = full_pipeline.named_transformers_["cat"].named_steps["encoder"]
cat_one_hot = cat_encoder.get_feature_names_out(categorical_features)
all_features = numeric_features + list(cat_one_hot)

feature_importances = pd.Series(forest.feature_importances_, index=all_features)
feature_importances = feature_importances.sort_values(ascending=False)
print("Feature importances:\n", feature_importances)

# %%
# Grid Search for Random Forest
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_features": ["auto", "sqrt", 0.5],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

forest_grid = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    forest_grid, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1
)
grid_search.fit(X_train_prepared, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best RMSE:", np.sqrt(-grid_search.best_score_))

# %%
# Train final model with best parameters
best_forest = RandomForestRegressor(**grid_search.best_params_, random_state=42)
best_forest.fit(X_train_prepared, y_train)

y_test_pred = best_forest.predict(X_test_prepared)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"Test RMSE: {rmse_test:.2f}")

# %%
# Test prediction with new data
new_client = pd.DataFrame(
    [
        {
            "age": 19,
            "bmi": 23.45,
            "children": 0,
            "sex": "male",
            "smoker": "no",
            "region": "southeast",
        }
    ]
)

new_client_prepared = full_pipeline.transform(new_client)
predicted_insurance = best_forest.predict(new_client_prepared)
print(f"Predicted insurance cost: {predicted_insurance[0]:.2f}")

# %%
# Save model and pipeline
os.makedirs("model", exist_ok=True)
with open("model/best_forest_model.pkl", "wb") as f:
    pickle.dump(best_forest, f)
with open("model/full_pipeline.pkl", "wb") as f:
    pickle.dump(full_pipeline, f)
