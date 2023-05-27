from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/predict", methods = ["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    else:
        data = CustomData(
            fixed_acidity = float(request.form.get("fixed_acidity")),
            volatile_acidity = float(request.form.get("volatile_acidity")),
            citric_acid = float(request.form.get("citric_acid")),
            residual_sugar= float(request.form.get("residual_sugar")),
            chlorides = float(request.form.get("chlorides")),
            free_sulfur_dioxide = float(request.form.get("free_sulfur_dioxide")),
            sulfur_dioxide = float(request.form.get("sulfur_dioxide")),
            density = float(request.form.get("density")),
            pH = float(request.form.get("pH")),
            sulphates = float(request.form.get("sulphates")),
            alcohol = float(request.form.get("alcohol"))
        )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)
        
        results = pred
        
        return render_template("result.html", final_result=results)
        
            
if __name__ == "__main__":
    app.run(host = "0.0.0.0", debug= True)