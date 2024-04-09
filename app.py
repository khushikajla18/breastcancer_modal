from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))


@app.route('/')
def hello_world():
    return render_template('main.html')


@app.route('/predict', methods=['POST','GET'])
def predict():
     int_features=[int(x) for x in request.form.values()]
     final=[np.array(int_features)]
     print(int_features)
     print(final)
     prediction=model.predict_proba(final)
     output='{0:.{1}f}'.format(prediction[0][1], 2)

     if output>str(0):
        return render_template('main.html',pred='The Breast cancer is Malignant {}'.format(output),bhai="Go to the hospital asap!")
     else:
        return render_template('main.html',pred='The Breast Cancer is Benign {}'.format(output),bhai="You are Safe for now")



if __name__ == "__main__":
    app.run(debug=True)