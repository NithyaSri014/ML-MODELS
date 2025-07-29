import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.set_page_config(page_title="Coffee Prediction", layout="centered")

st.title("☕ Coffee Buy Prediction Based on Personal Conditions")

# Input data
weather = st.selectbox("Weather", ["Sunny", "Rainy", "Overcast"])
timeofday = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening"])
sleep = st.selectbox("Sleep Quality", ["Good", "Poor"])
mood = st.selectbox("Mood", ["Fresh", "Tired", "Energetic"])

# Raw training data
data = {
    'Weather': ['Sunny', 'Rainy', 'Overcast', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Rainy', 'Sunny', 'Rainy'],
    'TimeOfDay': ['Morning', 'Morning', 'Afternoon', 'Afternoon', 'Evening', 'Morning', 'Morning', 'Afternoon', 'Evening', 'Morning'],
    'SleepQuality': ['Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Good', 'Poor'],
    'Mood': ['Tired', 'Fresh', 'Tired', 'Energetic', 'Tired', 'Fresh', 'Tired', 'Tired', 'Energetic', 'Tired'],
    'BuyCoffee': ['Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

# Encode all categorical variables
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Train model
X = df.drop('BuyCoffee', axis=1)
y = df['BuyCoffee']
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

# Prepare input
input_data = {
    'Weather': label_encoders['Weather'].transform([weather])[0],
    'TimeOfDay': label_encoders['TimeOfDay'].transform([timeofday])[0],
    'SleepQuality': label_encoders['SleepQuality'].transform([sleep])[0],
    'Mood': label_encoders['Mood'].transform([mood])[0]
}
input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict"):
    result_encoded = model.predict(input_df)[0]
    result = label_encoders['BuyCoffee'].inverse_transform([result_encoded])[0]
    st.success(f"🧠 Prediction: The customer will {'✅ buy' if result == 'Yes' else '❌ not buy'} coffee.")

    # Show decision tree
    st.subheader("🧩 Decision Tree")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(model, feature_names=X.columns, class_names=label_encoders['BuyCoffee'].classes_, filled=True)
    st.pyplot(fig)

    # Explain decision path
    st.subheader("🗺️ Decision Path (Node Traversal):")
    node_indicator = model.decision_path(input_df)
    feature = model.tree_.feature
    threshold = model.tree_.threshold

    for node_id in node_indicator.indices:
        if feature[node_id] != -2:
            fname = X.columns[feature[node_id]]
            fval = input_df.iloc[0, feature[node_id]]
            thresh = threshold[node_id]
            decision = "Yes" if fval <= thresh else "No"
            st.write(f"Node {node_id}: {fname} (value: {fval}) <= {thresh:.2f}? → {decision}")

