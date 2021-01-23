from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def index():
       return render_template("index.html") 

@app.route("/home")
def home():
    return render_template("index.html")



# @app.route('/github')
# def Github():
#     return render_template("https://github.com/")


@app.route('/predict1',methods=['POST'])
def predict1():

    feature1 = request.form['sepal_length']
    feature2 = request.form['sepal_width']
    feature3 = request.form['petal_length']
    feature4 = request.form['petal_width']


    li = [feature1, feature2, feature3, feature4]

    integer_ = [float(i) for i in li]
    final_features = [np.array(integer_)]
    predict_value = model.predict(final_features)
    return render_template('index.html', prediction_text_Algo1= f' The flower was {predict_value}')

if __name__ == "__main__":
    app.run(debug=True)

