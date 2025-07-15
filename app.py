import streamlit as st
import pandas as pd
import joblib
import json

# Load model
model = joblib.load("models/trained/reaction_outcome_classifier.pkl")

# Load reverse mapping
with open("mapper/reverse_category_mappings.json", "r") as f:
    reverse_map = json.load(f)

# Invert category maps: user input → encoded values
category_map = {
    feature: {v: int(k) for k, v in reverse_map[feature].items()}
    for feature in reverse_map
}

# Reverse reaction_outcome for decoding model predictions
reaction_outcome_decode = reverse_map["reaction_outcome"]
reaction_outcome_decode = {int(k): v for k, v in reaction_outcome_decode.items()}

# Streamlit UI
st.title("💊 Predict Drug Reaction Outcome")
st.markdown("Fill in patient and drug details to get a prediction.")

# Input form
age = st.number_input("Age", min_value=0, max_value=120, value=30)
sex = st.selectbox("Sex", list(category_map["sex"].keys()))
country = st.selectbox("Country", sorted(category_map["country"].keys()))
reaction = st.selectbox("Reaction", sorted(category_map["reaction"].keys()))
drug = st.selectbox("Drug", sorted(category_map["drug"].keys()))
age_group = st.selectbox("Age Group", category_map["age_group"].keys())

# Predict
if st.button("Predict"):
    try:
        # Encode input values
        input_dict = {
            "age": age,
            "sex": category_map["sex"][sex],
            "country": category_map["country"][country],
            "reaction": category_map["reaction"][reaction],
            "drug": category_map["drug"][drug],
            "age_group": category_map["age_group"][age_group]
        }

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]

        # Decode prediction
        readable_outcome = reaction_outcome_decode.get(prediction, "Unknown")

        st.success(f"🎯 Predicted Reaction Outcome: **{readable_outcome}**")

    except KeyError as e:
        st.error(f"❌ Input value not recognized: {e}")
