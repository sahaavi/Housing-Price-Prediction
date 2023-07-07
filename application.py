from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app=application
# route for home page
@app.route('/')
def index():
    return render_template('index.html')

# route for result page
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        data = CustomData(
            longitude=float(request.form['longitude']),
            latitude=float(request.form['latitude']),
            housing_median_age=float(request.form['housing_median_age']),
            total_rooms=float(request.form['total_rooms']),
            total_bedrooms=float(request.form['total_bedrooms']),
            population=float(request.form['population']),   
            households=float(request.form['households']),
            median_income=float(request.form['median_income']),
            ocean_proximity=request.form['ocean_proximity']
        )
        # convert data to dataframe
        df = data.to_df()
        # predict
        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)
        # render result
        return render_template('predict.html', prediction=prediction[0])
    
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8080)