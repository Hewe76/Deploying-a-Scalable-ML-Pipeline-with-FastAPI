import json
import requests

# Send a GET request to root

r = requests.get("http://127.0.0.1:8000")

# Print GET response
print("Status Code:", r.status_code)
print("Result:", r.json()["message"])  # should print: Welcome to the Income Prediction API!


# Prepare sample input for POST
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST request to /predict/
r = requests.post("http://127.0.0.1:8000/predict/", json=data)

# Print POST response
print("Status Code:", r.status_code)
print("Result:", r.json()["result"])  # should print: <=50K or >50K
