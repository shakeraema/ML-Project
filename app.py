from flask import Flask , request,render_template
import pickle
import numpy as np
import pandas as pd

app= Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('data', [])

        if len(features) != 22:
            return jsonify({"prediction": "Error: Please provide exactly 22 inputs."}), 400

        np_data = np.asarray(features, dtype=np.float32)
        prediction = model.predict(np_data.reshape(1, -1))

        output = "This person has Parkinson's Disease." if prediction[0] == 1 else "This person does not have Parkinson's Disease."

        return jsonify({"prediction": output})
        
    except Exception as e:
        return jsonify({"prediction": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
     app.run(debug=True)
