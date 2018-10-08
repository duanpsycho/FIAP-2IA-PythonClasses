from flask import Flask
from flask_restful import Api, request
from sklearn.externals import joblib

app = Flask("api-ml")
api = Api(app)


@app.route("/hello", methods=["GET"])
def hello():
    return "Hello World"


@app.route("/predict", methods=["POST"])
def predict():
    retorno = {}
    temp_max = request.values["temp_max"]
    precipt = request.values["precipt"]
    weekend = request.values["weekend"]

    predicted = [[int(temp_max), int(precipt), int(weekend)]]

    model = joblib.load("models/linear_regression_v1.bin")

    resultado = model.predict(predicted)

    retorno = resultado[0]

    return retorno


app.run(host="0.0.0.0", port=8080, debug=True)
