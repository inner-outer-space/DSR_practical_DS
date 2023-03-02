import streamlit as st
import pandas as pd


# make any grid with a function
def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid

mygrid = make_grid(2,2)


# Loading data
train = pd.read_csv("./data/training_data.csv")

column_names = category_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

for i, column in enumerate(column_names): 
    target_bins = train.loc[:, column].value_counts()
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.write('Graph for column:', column)
        with col2:
            st.bar_chart(target_bins)


#st.write("Here we create data using a table:")
#st.write(train)
#target_bins = train.loc[:, 'Churn'].value_counts()
#st.bar_chart(target_bins)
#mygrid[0][1].st.bar_chart(target_bins)

#target_bins = pd.cut(train['tenure'], bins=5, labels=['0-20', '21-40', '41-60', '61-80', '81-100'])
#st.bar_chart(target_bins)
#st.write("Here we create data using a table:")
#st.write(pd.DataFrame({
#    'first column': [1, 2, 3, 4],
#    'second column': [10, 20, 30, 40]
#}))
