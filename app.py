from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomException
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for a home page

# @app.route('/Home')
# def app_homepage():


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            course_category=request.form.get('CourseCategory'),
            time_spent_on_course=float(request.form.get('TimeSpentOnCourse')),
            number_of_video_watched=int(request.form.get('NumberOfVideosWatched')),
            number_of_quiz=int(request.form.get('NumberOfQuizzesTaken')),
            quiz_scores=float(request.form.get('QuizScores')),
            completion_rate=float(request.form.get('CompletionRate')),
            device_type=float(request.form.get('DeviceType'))

        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html', results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0")