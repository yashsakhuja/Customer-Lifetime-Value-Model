import streamlit as st
import pandas as pd
import joblib
from lifetimes.plotting import plot_history_alive
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import lifetimes
from lifetimes.utils import calculate_alive_path


st.title('AND CLTV Application')

mod_sum= pd.read_csv('model_summary_cltv.csv')
df_2022 = pd.read_csv('df_2022.csv')


ggf = joblib.load('model_ggf')
bgf = joblib.load('model_bgf')

# Take Input for Customer ID from the user

cust_ID = st.selectbox('Select a Customer ID:', mod_sum['Customer ID'])


customer_orders = df_2022[df_2022['Customer ID'] == str(cust_ID)]
customer_orders = customer_orders.drop(customer_orders.columns[0], axis=1)
customer_orders = customer_orders.reset_index(drop=True)

st.header("Customer Transactions in 2022")

customer_orders

st.divider()

st.header("Predictions")

# Input Number of days to predict from the user
t = st.number_input("Prediction Period (days)",min_value=1,max_value=365,value=150)

individual = mod_sum.loc[mod_sum['Customer ID'] == cust_ID]

prediction=bgf.predict(t,
            individual['frequency'],
            individual['recency'],
            individual['T'])
prediction = prediction.reset_index(drop=True)

time_t = int(round((t/30),0))

total_cltv= ggf.customer_lifetime_value(bgf,individual['frequency'],
                                                       individual['recency'],
                                                       individual['T'],
                                                       individual['monetary_value'],
                                                       time=time_t, #lifetime in months
                                                       freq='D', #frequency of data in the dataset
                                                       discount_rate=0.1 #parameter
                                                       )
total_cltv = total_cltv.reset_index(drop=True)

st.write("The predicted number of transactions for the selected customer in the upcoming",t,"days is: ", round(prediction[0],0))



st.write("The predicted total order values of the selected customer in the upcoming",t,"days is: £",round(total_cltv[0],0))



#--------Profit Margin----------
profit_margin = st.slider("Select Profit Margin", min_value=0.0, 
max_value=100.0, step=0.1, format="%.1f%%",value=20.0)

profit_margin = profit_margin/100

individual['Auto_CLV'] = individual['predicted_clv'] * profit_margin
#--------Profit Margin----------


total_cltv_profit = round(sum(individual['Auto_CLV']),0)
profit_margin = profit_margin*100


st.write("The predicted total value from this customer in the upcoming",t,"days at ", profit_margin,"% profit rate is:", f'£ {total_cltv_profit:.2f}')

st.divider()

# Concatenate text and value into a single string
subheader_text = "CLTV: £" + str(total_cltv_profit)

# Display the subheader
st.subheader(subheader_text)

st.divider()

st.header("Plotting Customer's Alive/Dead Probabilities")

days_since_birth = st.number_input("Days from first transaction",min_value=1,max_value=1000,value=150)


# Convert the column to datetime
customer_orders['Extracted Invoice Date'] = pd.to_datetime(customer_orders['Extracted Invoice Date'], format='%Y-%m-%d')

# Change datatype to datetime64[D]
customer_orders['Extracted Invoice Date'] = customer_orders['Extracted Invoice Date'].dt.floor('D')


p_alive_data=lifetimes.utils.calculate_alive_path(bgf, customer_orders, "Extracted Invoice Date", days_since_birth, freq='D')


first_tran_date = min(customer_orders['Extracted Invoice Date'])

date_lim= max(len(p_alive_data),days_since_birth)

# Generating dates
date_range = [first_tran_date + timedelta(days=i) for i in range(date_lim)]

# Creating a DataFrame for plotting
data = pd.DataFrame({'Date': date_range, 'Probability': p_alive_data})
# Drop the index
data.reset_index(drop=True, inplace=True)


fig, ax = plt.subplots()
ax.plot(data['Date'], data['Probability'], marker='', linestyle='-', linewidth=1.2) # Make line thinner
ax.set_xlabel('Date')
ax.set_ylabel('P_alive')
ax.set_title('P_Alive History')
ax.set_ylim(0, 1)  # Set y-axis limits to 0 and 1

for date in customer_orders['Extracted Invoice Date']:
    ax.axvline(x=date, color='red', linestyle='--',linewidth=0.8)

plt.xticks(rotation=45)
st.pyplot(fig)










