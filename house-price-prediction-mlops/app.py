from flask import Flask, render_template, request

from src.prediction import HousePricePredictor

app = Flask(__name__)


def _form_payload(form) -> dict:
    numeric_fields = ["area", "bedrooms", "bathrooms", "stories", "parking", "age"]
    payload = {field: float(form[field]) for field in numeric_fields}
    for field in [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
        "furnishingstatus",
        "city_zone",
    ]:
        payload[field] = form[field]
    return payload


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            predictor = HousePricePredictor()
            price = predictor.predict(_form_payload(request.form))
            prediction = f"{price:,.0f}"
        except Exception as exc:
            error = str(exc)
    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
