from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
# from sklearn.preprocessing import OneHotEncoder


app = Flask(__name__)
file_path = '/workspaces/heart/models/cv.joblib'

try:
    # Load data from .joblib file
    data = joblib.load(file_path)

    # Handle the loaded data appropriately
    # Your processing logic here
    
    print("Data loaded successfully:", data)

except FileNotFoundError:
    print("File not found: ", file_path)

except Exception as e:
    print("Error loading the file:", str(e))
# model = joblib.load(open('/workspaces/heart/models/cv.joblib',encoding='utf-8'))
# Define a route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')  # Replace 'your_form.html' with the path to your HTML file

# Define a route to handle the form submission
@app.route('/submit', methods=['POST'])
def post():
    form_data = {}  # Create an empty dictionary to store form data

    general_health = request.form.get('general_health')
    checkup_frequency = request.form.get('checkup_frequency')
    exercise = request.form.get('exercise')
    skin_cancer = request.form.get('skin_cancer')
    other_cancer = request.form.get('other_cancer')
    Depression = request.form.get('Depression')
    diabetes = request.form.get('diabetes')
    Arthritis = request.form.get('Arthritis')
    sex = request.form.get('Sex')
    age_category = request.form.get('Age_Categroy')
    # Access other form fields in a similar manner

    height = request.form.get('height')
    weight = request.form.get('weight')
    bmi = request.form.get('bmi')

    smoking_history = request.form.get('smoking_history')
    alcohol_consumption = request.form.get('alcohol_consumption')
    fruit_consumption = request.form.get('fruit_consumption')
    green_vegetable_consumption = request.form.get('green_vegetable_consumption')
    fried_potato_consumption = request.form.get('fried_potato_consumption')
    # Access other form fields in a similar manner
    print(type(bmi))
    # Store form data in the dictionary
    form_data['General_Health'] = general_health
    form_data['Checkup'] = checkup_frequency
    form_data['Exercise'] = exercise
    form_data['Skin_Cancer'] = skin_cancer
    form_data['Other_Cancer'] = other_cancer
    form_data['Depression'] = Depression
    form_data['Diabetes'] = diabetes
    form_data['Arthritis'] = Arthritis
    form_data['Sex'] = sex
    form_data['Age_Category'] = age_category

    # Store other form fields in the dictionary as needed

    form_data['Height_(cm)'] = float(height)
    form_data['Weight_(kg)'] = float(weight)
    form_data['BMI'] = float(bmi)

    form_data['Smoking_History'] = smoking_history
    form_data['Alcohol_Consumption'] = float(alcohol_consumption)
    form_data['Fruit_Consumption'] = float(fruit_consumption)
    form_data['Green_Vegetables_Consumption'] = float(green_vegetable_consumption)
    form_data['FriedPotato_Consumption'] = float(fried_potato_consumption)

    df = pd.DataFrame([form_data])
    # print(df)
    df = convert(df,sex)
    pred = data.predict(df)
    if pred == 1:
        return "heart disease found"
    return "Not found"
    # print(f"value is : {pred}")
    # # Store other form fields in the dictionary as needed

    # # Print the form data dictionary to the terminal
    # # print(df.iloc[0].to_dict())


    return pred

def convert(data,gender):
    data['BMI_Category'] = pd.cut(data['BMI'], bins=[0, 18.5, 24.9, 29.9, np.inf], labels=['Underweight', 'Normal weight', 'Overweight', 'Obesity'])

    # Health Checkup Frequency
    checkup_mapping = {'Within the past year': 4, 'Within the past 2 years': 2, 'Within the past 5 years': 1, '5 or more years ago': 0.2, 'Never': 0}
    data['Checkup_Frequency'] = data['Checkup'].replace(checkup_mapping)

    # Lifestyle Score
    exercise_mapping = {'Yes': 1, 'No': 0}
    smoking_mapping = {'Yes': -1, 'No': 0}
    data['Lifestyle_Score'] = data['Exercise'].replace(exercise_mapping) - data['Smoking_History'].replace(smoking_mapping) + data['Fruit_Consumption']/10 + data['Green_Vegetables_Consumption']/10 - data['Alcohol_Consumption']/10

    # Healthy Diet Score
    data['Healthy_Diet_Score'] = data['Fruit_Consumption']/10 + data['Green_Vegetables_Consumption']/10 - data['FriedPotato_Consumption']/10
    data['Smoking_Alcohol'] = data['Smoking_History'].replace(smoking_mapping) * data['Alcohol_Consumption']
    data['Checkup_Exercise'] = data['Checkup_Frequency'] * data['Exercise'].replace(exercise_mapping)

    # Ratio of Height to Weight
    data['Height_to_Weight'] = data['Height_(cm)'] / data['Weight_(kg)']

    # Fruit and Vegetables Consumption Interaction
    data['Fruit_Vegetables'] = data['Fruit_Consumption'] * data['Green_Vegetables_Consumption']

    # Healthy_Diet_Lifestyle Interaction
    data['HealthyDiet_Lifestyle'] = data['Healthy_Diet_Score'] * data['Lifestyle_Score']

    # Alcohol_FriedPotato Interaction
    data['Alcohol_FriedPotato'] = data['Alcohol_Consumption'] * data['FriedPotato_Consumption']

    diabetes_mapping = {
        'No': 0, 
        'No, pre-diabetes or borderline diabetes': 0, 
        'Yes, but female told only during pregnancy': 1,
        'Yes': 1
    }
    data['Diabetes'] = data['Diabetes'].map(diabetes_mapping)

    # One-hot encoding for Sex
    # categorical_cols = ['Sex']
    # encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    # encoder.fit(data[categorical_cols])
    # encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    # print(encoded_cols)
    # data[encoded_cols] = encoder.transform(data[categorical_cols])
    # data['Sex'].astype(str)
    # gender = data ['Sex']
    # data = pd.get_dummies(data, columns=['Sex'])
    if gender == 'Female':
        data['Sex_Female'] = 1
        data['Sex_Male'] = 0
    else:
        data['Sex_Female'] = 0
        data['Sex_Male'] = 1

    # Convert remaining categorical variables with "Yes" and "No" values to binary format for correlation computation
    binary_columns = ['Skin_Cancer', 'Other_Cancer', 'Depression', 'Arthritis', 'Smoking_History','Exercise']

    for column in binary_columns:
        data[column] = data[column].map({'Yes': 1, 'No': 0})
        
    # Ordinal encoding for General_Health, Age_Category,BMI_Category
    general_health_mapping = {
        'Poor': 0,
        'Fair': 1,
        'Good': 2,
        'Very Good': 3,
        'Excellent': 4
    }
    data['General_Health'] = data['General_Health'].map(general_health_mapping)

    bmi_mapping = {
        'Underweight': 0,
        'Normal weight': 1,
        'Overweight': 2,
        'Obesity': 3
    }

    data['BMI_Category'] = data['BMI_Category'].map(bmi_mapping).astype(int)

    age_category_mapping = {
        '18-24': 0,
        '25-29': 1,
        '30-34': 2,
        '35-39': 3,
        '40-44': 4,
        '40-44': 4,
        '45-49': 5,
        '50-54': 6,
        '55-59': 7,
        '60-64': 8,
        '65-69': 9,
        '70-74': 10,
        '75-79': 11,
        '80+': 12
    }
    data['Age_Category'] = data['Age_Category'].map(age_category_mapping)    

    data = data.drop(["Checkup","Sex"],axis=1)
    return data
if __name__ == '__main__':
    app.run(debug=True)
