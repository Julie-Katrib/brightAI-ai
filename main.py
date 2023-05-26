from flask import Flask,request,jsonify
import numpy as np
import pickle

model_treatment = pickle.load(open('decision_tree_model_treatment','rb'))
model_diagnosis = pickle.load(open('decision_tree_model_diagnosis','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    age = request.form.get('age')
    gender = request.form.get('gender')
    medical_history = request.form.get('medical_history')
    dental_history = request.form.get('dental_history')
    location_of_pain = request.form.get('location_of_pain')
    pain_severity = request.form.get('pain_severity')
    symptoms = request.form.get('symptoms')
    #diagnosis = request.form.get('diagnosis')
    patient_id = request.form.get('patientId')
    input_query_diagnosis = np.array([[patient_id,age,gender,medical_history,dental_history,location_of_pain,pain_severity,symptoms]])

    diagnosis_result = model_diagnosis.predict(input_query_diagnosis)[0]

    input_query_treatment = np.concatenate((input_query_diagnosis, np.array([[diagnosis_result]])), axis=1)

    treatment_result = model_treatment.predict(input_query_treatment)[0]
    return jsonify({'diagnosis': str(diagnosis_result), 'treatment': str(treatment_result)})

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug=False, host='0.0.0.0')
