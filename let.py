from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression  # Added import for Logistic Regression
from sklearn.metrics import accuracy_score
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and train the diabetes model
diabetes_dataset = pd.read_csv('diabetes.csv')

# Separating the data and labels
x_diabetes = diabetes_dataset.drop(columns='Outcome', axis=1)
y_diabetes = diabetes_dataset['Outcome']

# Standardizing the data
scaler = StandardScaler()
scaler.fit(x_diabetes)
standardized_data = scaler.transform(x_diabetes)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    standardized_data, y_diabetes, test_size=0.2, stratify=y_diabetes, random_state=2
)

# Training the model
diabetes_classifier = svm.SVC(kernel='linear')
diabetes_classifier.fit(X_train, Y_train)


@app.route("/")
def index():
    return "Hello, World! The Diabetes Prediction Model is Ready."


@app.route("/predict", methods=["POST"])
def predict_diabetes():
    try:
        # Parse the JSON data from the POST request
        input_data = request.get_json()

        # Ensure all required fields are present
        required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        for field in required_fields:
            if field not in input_data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Convert input data to a NumPy array
        input_values = [input_data[field] for field in required_fields]
        data_into_array = np.asarray(input_values)

        # Reshape the data for prediction
        reshaped_data = data_into_array.reshape(1, -1)

        # Standardize the input data
        std_data = scaler.transform(reshaped_data)

        # Make a prediction
        output = diabetes_classifier.predict(std_data)

        # Return the prediction result
        result = "Diabetic" if output[0] == 1 else "Non-Diabetic"
        return jsonify({"prediction": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Load the heart disease dataset
# Load and preprocess the data
heart_data = pd.read_csv('heart.csv')  # Ensure 'heart.csv' is in the same directory or provide the full path
x_heart = heart_data.drop(columns='target', axis=1)
y_heart = heart_data['target']

# Split the data for training
x_train_heart, x_test_heart, y_train_heart, y_test_heart = train_test_split(
    x_heart, y_heart, test_size=0.2, stratify=y_heart, random_state=2
)

# Train the Logistic Regression model
heart_model = LogisticRegression(solver='liblinear')
heart_model.fit(x_train_heart, y_train_heart)

@app.route("/heart", methods=["POST"])
def predict_heart_disease():
    try:
        # Parse the JSON data from the request
        data = request.get_json()
        print("Received data:", data)

        # Extract required fields from the input
        input_data = {
            "age": float(data["age"]),
            "sex": int(data["sex"]),
            "cp": int(data["cp"]),
            "trestbps": float(data["trestbps"]),
            "chol": float(data["chol"]),
            "fbs": int(data["fbs"]),
            "restecg": int(data["restecg"]),
            "thalach": float(data["thalach"]),
            "exang": int(data["exang"]),
            "oldpeak": float(data["oldpeak"]),
            "slope": int(data["slope"]),
            "ca": int(data["ca"]),
            "thal": int(data["thal"])
        }

        # Convert input data to DataFrame with the same feature names as training
        input_df = pd.DataFrame([input_data])

        # Perform prediction
        prediction = heart_model.predict(input_df)

        # Interpret the prediction
        result = "Person has heart disease" if prediction[0] == 1 else "Person does not have heart disease"

        # Return the prediction result
        return jsonify({"prediction": result}), 200

    except KeyError as e:
        # Handle missing fields
        return jsonify({"error": f"Missing field: {str(e)}"}), 400
    except Exception as e:
        # Log the error for debugging
        print(f"Error: {str(e)}")
        return jsonify({"error": "An error occurred during prediction."}), 500



if __name__ == "__main__":
    app.run(debug=True)
