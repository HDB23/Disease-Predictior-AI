# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# import pickle
#
# # Load the data
# data_path = "Disease_symptom_and_patient_profile_dataset.csv"
# data = pd.read_csv(data_path)
#
# # Prepare the features and target variable
# X = data.drop(columns=["Disease", "Outcome Variable"])  # Drop the disease and outcome variable columns
# y = data["Disease"]
#
# # Encode the Disease to numerical values
# disease_encoder = LabelEncoder()
# y = disease_encoder.fit_transform(y)
#
# # Encoding categorical variables
# le_dict = {}
# for column in X.columns:
#     if X[column].dtype == 'object':
#         le = LabelEncoder()
#         X[column] = le.fit_transform(X[column])
#         le_dict[column] = le
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Train the model
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
#
# # Save the trained model, label encoders, and disease encoder
# with open("model.pkl", "wb") as model_file:
#     pickle.dump(model, model_file)
#
# with open("label_encoders.pkl", "wb") as encoders_file:
#     pickle.dump(le_dict, encoders_file)
#
# with open("disease_encoder.pkl", "wb") as disease_encoder_file:
#     pickle.dump(disease_encoder, disease_encoder_file)
#
# print("Model, label encoders, and disease encoder saved successfully!")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the data
data_path = "Disease_symptom_and_patient_profile_dataset.csv"
data = pd.read_csv(data_path)

# Prepare the features and target variable
X = data.drop(columns=["Outcome Variable", "Disease"])
y = data["Disease"]

# Encode the Disease to numerical values
disease_encoder = LabelEncoder()
y = disease_encoder.fit_transform(y)

# Encoding categorical variables
le_dict = {}
for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        le_dict[column] = le

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model, label encoders, and disease encoder
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("label_encoders.pkl", "wb") as encoders_file:
    pickle.dump(le_dict, encoders_file)

with open("disease_encoder.pkl", "wb") as disease_encoder_file:
    pickle.dump(disease_encoder, disease_encoder_file)

print("Model, label encoders, and disease encoder saved successfully!")

