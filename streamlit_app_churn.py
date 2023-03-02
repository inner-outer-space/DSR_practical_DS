import streamlit as st
import pandas as pd
import rachels_predictions as rp
import rachels_model_training as rm
#import predictions as rp
#import model_training as rm


# title
st.title("Customer Churn Prediction")
st.subheader("Enter the customer's data")

# Create the customer data input form 
with st.form("form1"):
    gender = st.selectbox('gender?',('Male', 'Female'))
    senior_citizen = st.selectbox('Senior Citizen?',('Yes', 'No'))
    partner = st.selectbox('Partner?',('Yes', 'No'))
    dependents = st.selectbox('Dependents?',('Yes', 'No'))
    tenure = st.slider('What is the customer\'s tenure? :', min_value=0, max_value=72, value=20)
    phone_service = st.selectbox('Phone Service?',('Yes', 'No'))
    multiple_lines = st.selectbox('Multiple Lines?',('Yes', 'No','No phone service'))
    internet_service = st.selectbox('Internet Service?',('Fiber optic', 'DSL','No'))
    online_security = st.selectbox('Online Security?',('Yes', 'No','No phone service'))
    online_backup = st.selectbox('Online Backup?',('Yes', 'No','No phone service'))
    device_protection = st.selectbox('Device Protection?',('Yes', 'No','No phone service'))
    tech_support = st.selectbox('Tech Support?',('Yes', 'No','No phone service'))
    streaming_tv = st.selectbox('Streaming TV?',('Yes', 'No','No phone service'))
    streaming_movies = st.selectbox('Streaming Movies?',('Yes', 'No','No phone service'))
    contract = st.selectbox('Contract?',('Month-to-month','Two year','One year'))
    paperless_billing = st.selectbox('Paperless Billing?',('Yes', 'No'))
    payment_method = st.selectbox('Payment Method?',('Electronic check', 'Mailed check','Bank transfer (automatic)','Credit card (automatic)'))
    monthly_charges = st.slider("What is the customer's monthly charge? :", min_value=0, max_value=118, value=50)
    total_charges = st.slider('What is the total charge of the customer? :', min_value=0, max_value=8600, value=2000)
    st.form_submit_button()

# Create a dictionary of the customer data
input_dict = {'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    }

# Create a dataframe of the customer data
column_order = list(input_dict.keys())
customer_data = pd.DataFrame([input_dict], index=[0], columns=column_order)
#customer_data = customer_data.reset_index(drop=True, inplace=True)
#customer_data = pd.DataFrame([input_dict])

# Display the customer data
st.write("Here is the input data:")
st.table(customer_data)

# Add the missing columns to the customer data
customer_data['Churn'] = ''
customer_data['customerID'] = ''

# Convert the SeniorCitizen column to a binary value
if customer_data['SeniorCitizen'].values[0] == 'Yes':   
    customer_data['SeniorCitizen'] = 1
else:
    customer_data['SeniorCitizen'] = 0  

#Generate the prediction for the customer
if st.button("Predict Churn"):
    pred = rp.generate_predictions(customer_data)
    st.write(pred)
    if bool(pred):
        st.text("Customer will churn!")
    else:
        st.text("Customer not predicted to churn")