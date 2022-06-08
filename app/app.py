
def main():
    import pickle
    import numpy as np
    import pandas as pd
    import streamlit as st
    
    # loading the model
    models_path = '../models/'
    model_name = models_path + 'finalized_model.sav'
    loaded_model = pickle.load(open(model_name, 'rb'))

    # loading the scaler
    scalers_path = '../scalers/'
    scaler_name = scalers_path + 'scaler4.pkl'
    loaded_scaler = pickle.load(open(scaler_name, 'rb'))

    # Main page #
    st.title("Prediction demo credit card acceptance")     


    reward_desired = st.selectbox('Reward', ('Air Miles','Cashback','Points')),

    if 'Air Miles' in reward_desired:
        reward_desired = 0
    elif 'Cashback' in reward_desired:
        reward_desired = 1
    elif 'Points' in reward_desired:
        reward_desired = 2

    mailer_type = st.selectbox('Mailer type', ('Letter', 'Postcard'))

    if 'Letter' in mailer_type:
        mailer_type = 0
    elif 'Postcard' in mailer_type:
        mailer_type = 1

    income_level = st.selectbox('Income level', ('Low', 'Medium', 'High'))

    if 'Low' in income_level:
        income_level = 0
    elif 'Medium' in income_level:
        income_level = 1
    elif 'High' in income_level:
        income_level = 2


    open_accounts = st.slider('open accounts', 0, 10, 0)

    overdraft_protection = st.selectbox('overdraft protection', ('Yes', 'No'))

    if 'Yes' in overdraft_protection:
        overdraft_protection = 1
    elif 'No' in overdraft_protection:
        overdraft_protection = 0



    credit_rating = st.selectbox('credit rating' , ('Low', 'Medium', 'High'))

    if 'Low' in credit_rating:
        credit_rating = 0
    elif 'Medium' in credit_rating:
        credit_rating = 1
    elif 'High' in credit_rating:
        credit_rating = 2


    credit_cards_held = st.slider('credit cards held', 0, 10, 0)
    homes_owded = st.number_input('owned homes', min_value=0, max_value=1000, value=0,step=1)
    household_size = st.number_input('household size', min_value=0, max_value=100,step=1, value=0)
    owner_home = st.selectbox('home own', ('Yes', 'No'))

    if 'Yes' in owner_home:
        owner_home = 1
    elif 'No' in owner_home:
        owner_home = 0


    balanced_q1 = st.number_input('average balanced Q1',step=10)
    balanced_q2 = st.number_input('average balanced Q2',step=10)
    balanced_q3 = st.number_input('average balanced Q3',step=10)
    balanced_q4 = st.number_input('average balanced Q4',step=10)
    balance_average = st.text('balance average pending computation')

    def average(x):
        avg = sum(x) / len(x)
        return avg


    if st.button("Average Balance"):
        Z = {balanced_q1, balanced_q2, balanced_q3, balanced_q4}
        Z_avg = average(Z)
        balance_average.text("Average value computed " + str(Z_avg))



    Z = {balanced_q1, balanced_q2, balanced_q3, balanced_q4}
    Z_avg = average(Z)


    if st.button("PREDICT"):
    #encoding String variables

        X = pd.DataFrame({'Reward': [reward_desired],
                        'Mailer type': [mailer_type],
                        'Income Level': [income_level],
                        'Bank accounts opened': [open_accounts],
                        'Overdraft protection': [overdraft_protection],
                        'Credit Rating': [credit_rating],
                        'Credit cards held': [credit_cards_held],
                        'Homes Owned': [homes_owded],
                        'Household size': [household_size],
                        'Owned home': [owner_home],
                        'Avg Balance': [Z_avg],
            })

        X['Balance variability 2'] = abs((X.std(axis=1) - X.min(axis=1)) / (X.max(axis=1) - X.min(axis=1) / X.sum(axis=1)))
        
        # Scaling data
        X_scaled = loaded_scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        # Making predictions
        prediction = loaded_model.predict(X_scaled_df.values)
        
        # Displaying the prediction
        if (prediction == 1):
            st.success("Your client could accept the offer, 72% chance of true positive")
         
        else:
            st.success("Your client would not accept the offer")
            

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
