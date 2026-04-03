from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load and clean
df = pd.read_excel(r"E:\project_dv\paper1\mmc2.xlsx", sheet_name="Raw Data")
q_cols = df.columns[1:21].tolist()
df = df[q_cols].apply(lambda col: col.astype(str).str.strip().str.lower())

# One-Hot indexes (Q2–Q4, Q8, Q12–Q16, Q19)
onehot_cols_index = [1, 2, 3, 7, 11, 12, 13, 14, 15, 18]

# Ordinal indexes (Q5–Q7, Q9–Q11, Q17, Q18, Q20)
ordinal_cols_index = [4, 5, 6, 8, 9, 10, 16, 17, 19]

# Final Ordinal Categories (9 total)
ordinal_categories_final = [
    ["less than 2-year", "2-5 years", "5-10 years", "more than 10 years"],         # Q5
    ["less than 1 per day", "1-2 per day", "3-5 per day", "more than 5 per day"],  # Q6
    ["less than 1 hour", "1-3 hours", "3-5 hours", "more than 5 hours"],           # Q7
    ["less than 500", "500-2000", "2000-4000", "more than 4000"],                  # Q9
    ["few of them", "many of them", "most of them", "all of them"],                # Q10
    ["less than 5", "5-10", "10-20", "more than 20"],                              # Q11
    ["not at all", "sometimes", "always"],                                         # Q17
    ["never", "most of the times", "all the times"],                               # Q18
    ["no, i am not trying", "not trying", "trying to reduce the use",
     "yes, i am trying but can’t", "yes, i am trying and i have reduced using it", "trying to stop the use"]  # Q20
]
# Preprocessor (force dense output from OneHotEncoder)
preprocessor_final = ColumnTransformer(transformers=[
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols_index),
    ('ord', OrdinalEncoder(categories=ordinal_categories_final), ordinal_cols_index)
])
# Pipeline
pipeline_final = Pipeline(steps=[
    ('preprocess', preprocessor_final)
])

# Fit-transform
encoded_data_final = pipeline_final.fit_transform(df)

# Inspect
encoded_df_final = pd.DataFrame(encoded_data_final, columns=pipeline_final.named_steps['preprocess'].get_feature_names_out())

# print(encoded_df_final.head())

# Assuming encoded_data_final is ready (from pipeline)
# Load your target variable (example: UCLA)
df_social = pd.read_csv(r"E:\project_dv\paper1\social_df.csv")
y_ucla = df_social["ucla_score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(encoded_data_final, y_ucla, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "ucla_model.pkl")
print("✅ UCLA Model Saved!")
# Save the pipeline if you want to reuse preprocessing easily
joblib.dump(pipeline_final, "ucla_pipeline.pkl")
print("✅ Pipeline Saved!")

# PHQ-9
y_phq = df_social["PHQ9_score"]
model_phq = RandomForestRegressor(random_state=42)
model_phq.fit(X_train, y_phq.loc[y_train.index])  # Align index
joblib.dump(model_phq, "phq_model.pkl")
print("✅ PHQ-9 Model Saved!")

# GAD-7
y_gad = df_social["GAD7_score"]
model_gad = RandomForestRegressor(random_state=42)
model_gad.fit(X_train, y_gad.loc[y_train.index])  # Align index
joblib.dump(model_gad, "gad_model.pkl")
print("✅ GAD-7 Model Saved!")

#PSQI
y_psqi = df_social["PSQI_score"]
model_psqi = RandomForestRegressor(random_state=42)
model_psqi.fit(X_train, y_psqi.loc[y_train.index])  # Align index
joblib.dump(model_psqi, "psqi_model.pkl")
print("✅ PSQI Model Saved!")

