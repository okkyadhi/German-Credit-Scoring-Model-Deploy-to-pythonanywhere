#!/usr/bin/python3.6
from flask import Flask, request, jsonify #To handle the communication
import pandas as pd
# import sklearn.preprocessing
import pickle
# from sklearn.externals import joblib #To open imported model in the server
import sklearn.preprocessing as preprocessing

app = Flask(__name__)

model = pickle.load(open('model_lgb.pkl','rb'))

@app.route('/api',methods=['POST'])

def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Transform JSON into DataFrame
    data = pd.DataFrame.from_dict(data)

    def age_group(age):
        if age < 20:
            return 'Teenager'
        if age < 35:
            return 'Productive'
        if age < 50:
            return 'Mature'
        else :
             return 'Old'

    data['credit_per_person'] = data['credit_amount']/data['people_under_maintenance']
    data['age_group'] = data['age'].apply(age_group)

    # Create x, where x the 'scores' column's values as floats
    x = data[['credit_amount']].values.astype(float)

    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x)

    # Run the normalizer on the dataframe
    data['credit_amount_norm'] = pd.DataFrame(x_scaled)
    data['age_norm']= pd.DataFrame(min_max_scaler.fit_transform(data[['age']].values.astype(float)))
    data['duration_in_month']= pd.DataFrame(min_max_scaler.fit_transform(data[['duration_in_month']].values.astype(float)))

    temp = data.drop(['age','installment_as_income_perc','personal_status_sex','telephone'],axis=1)

    dataset_dummies = temp.select_dtypes(exclude=['int','int64','float64'])
    dataset_int = temp.select_dtypes(include = ['int','int64','float64'] )


    dataset_dummies_2 = pd.get_dummies(dataset_dummies, drop_first=True)
    data2 = pd.concat([dataset_int,dataset_dummies_2], axis = 1 )



    prediction = model.predict(data2)
    pred = [round(value).astype(int) for value in prediction]

    result = pd.DataFrame()
    # result['prob'] = y_pred
    result['pred'] = pred

    # Take the first value of prediction
    output =   result['pred'].tolist()

    return jsonify(output)

    
if __name__ == '__main__':
    try:
        app.run(port=5005, debug=True)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")