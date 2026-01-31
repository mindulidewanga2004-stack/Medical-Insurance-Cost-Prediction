from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load YOUR trained model
with open('insurance_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('newindex.html')

@app.route('/prediction', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        bmi = float(request.form['bmi'])
        children = float(request.form['children'])
        sex = request.form['sex']
        smoker = 1 if request.form['smoker'] == '1' else 0  # Match your dataset (yes=1, no=0)
        region = request.form['region']

        # EXACT column order from YOUR dataset
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'sex': [sex],
            'smoker': [smoker],
            'region': [region]
        })

        prediction = model.predict(input_data)[0]
        return render_template('newindex.html', 
                             prediction_text=f"💰 Estimated Insurance Cost: ${prediction:,.2f}",
                             show_summary=True,
                             summary_data=request.form)

    except Exception as e:
        return render_template('newindex.html', 
                             prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, port=5017)
