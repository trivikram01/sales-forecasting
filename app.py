from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the saved model and encoders
model = pickle.load(open('sales_model.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    features = [x for x in request.form.values()]
    
    # 1. Parse Inputs
    # User inputs: S1, L1, R1, 1 (Holiday), Yes (Discount), 2023-01-01 (Date)
    store = features[0]
    loc = features[1]
    region = features[2]
    holiday = int(features[3])
    discount = features[4]
    date_input = pd.to_datetime(features[5])
    
    # 2. Encode Strings to Numbers (using the saved encoders)
    # We use .transform() to convert "S1" -> 0, etc.
    store_enc = encoders['Store_Type'].transform([store])[0]
    loc_enc = encoders['Location_Type'].transform([loc])[0]
    region_enc = encoders['Region_Code'].transform([region])[0]
    discount_enc = encoders['Discount'].transform([discount])[0]
    
    # 3. Extract Date Features
    year = date_input.year
    month = date_input.month
    day = date_input.day
    dayofweek = date_input.dayofweek
    
    # 4. Combine into a list
    final_features = np.array([[store_enc, loc_enc, region_enc, holiday, discount_enc, year, month, day, dayofweek]])
    
    # 5. Predict
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text='Predicted Sales: ${}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)