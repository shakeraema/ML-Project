from flask import Flask , request,render_template
import pickle
import numpy as np
import pandas as pd

app= Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods = ['POST'])
def predict():
     input_text = request.form['text']
     input_text_sp = input_text.split(',')
     np_data = np.asarray(input_text_sp,dtype = np.float32)
     prediction = model.predict(np_data.reshape(1,-1))

     if prediction == 1 :
          output = "This person has a Parkinson Disease"
     else:
          output = "This person has no Parkinson Disease"

     return render_template("index.html", message=output)

if __name__ == "__main__":
     app.run(debug=True)