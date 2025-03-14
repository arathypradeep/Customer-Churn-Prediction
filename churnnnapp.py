import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# Load the trained churn prediction model
model = pickle.load(open('churnnn.pkl', 'rb'))

# Function to get user inputs
def user_input_parameters():
    st.sidebar.header("📋 Customer Information")

    state = st.sidebar.selectbox('🌍 State (0-51)', list(range(52)))
    area_code = st.sidebar.selectbox('📞 Area Code', [0, 1, 2])
    account_length = st.sidebar.number_input('📆 Account Length (days)', 0, 250)

    voice_plan = st.sidebar.radio('☎️ Voice Plan (Yes=1, No=0)', [0, 1])
    voice_messages = st.sidebar.number_input('💬 Voice Messages', 0, 50)

    intl_plan = st.sidebar.radio('🌎 International Plan (Yes=1, No=0)', [0, 1])
    intl_mins = st.sidebar.number_input('⏳ Intl Mins', 0.0, 50.0)
    intl_calls = st.sidebar.number_input('📲 Intl Calls', 0, 20)
    intl_charge = st.sidebar.number_input('💰 Intl Charge', 0.0, 20.0)

    day_mins = st.sidebar.number_input('🌞 Day Minutes', 0.0, 400.0)
    day_calls = st.sidebar.number_input('📞 Day Calls', 0, 200)
    day_charge = st.sidebar.number_input('💵 Day Charge', 0.0, 60.0)

    eve_mins = st.sidebar.number_input('🌙 Evening Minutes', 0.0, 400.0)
    eve_calls = st.sidebar.number_input('📱 Evening Calls', 0, 200)
    eve_charge = st.sidebar.number_input('💸 Evening Charge', 0.0, 60.0)

    night_mins = st.sidebar.number_input('🌃 Night Minutes', 0.0, 400.0)
    night_calls = st.sidebar.number_input('📴 Night Calls', 0, 200)
    night_charge = st.sidebar.number_input('💤 Night Charge', 0.0, 60.0)

    customer_calls = st.sidebar.number_input('📢 Customer Service Calls', 0, 10)

    data = {
        'state': state, 'area.code': area_code, 'account.length': account_length,
        'voice.plan': voice_plan, 'voice.messages': voice_messages,
        'intl.plan': intl_plan, 'intl.mins': intl_mins, 'intl.calls': intl_calls, 'intl.charge': intl_charge,
        'day.mins': day_mins, 'day.calls': day_calls, 'day.charge': day_charge,
        'eve.mins': eve_mins, 'eve.calls': eve_calls, 'eve.charge': eve_charge,
        'night.mins': night_mins, 'night.calls': night_calls, 'night.charge': night_charge,
        'customer.calls': customer_calls
    }

    return pd.DataFrame(data, index=[0])

# Dashboard Title
st.title("🚀 Churn Prediction Dashboard")
st.markdown("📊 **Predict customer churn and improve retention strategies!**")

df = user_input_parameters()

if st.button('🔮 Predict Churn'):
    # Ensure the input data columns match the model’s expected feature names
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Try predicting probabilities safely
    try:
        probs = model.predict_proba(df)

        if probs.shape[1] == 2:  # If two probability values exist (No Churn, Churn)
            pred_prob = probs[0][1]  # Churn probability
        else:
            pred_prob = 0  # Default if not available

        # Display Prediction Result
        st.subheader('📌 Prediction Result')
        result_text = '❌ **Churn (Yes)**' if pred_prob >= 0.5 else '✅ **Loyal Customer (No)**'
        st.markdown(f"<h2 style='text-align: center; color: {'red' if pred_prob >= 0.5 else 'green'}'>{result_text}</h2>", unsafe_allow_html=True)

        # Display Churn Probability with Progress Bars
        st.subheader('📊 Predicted Probability')
        st.progress(pred_prob)  # Simple visual indicator
        st.success(f'🔵 No Churn Probability: {100 - pred_prob*100:.2f}%')
        st.error(f'🔴 Churn Probability: {pred_prob*100:.2f}%')

        # Create a Gauge Meter using Plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_prob * 100,
            title={'text': "Churn Probability (%)", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "red" if pred_prob >= 0.5 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))

        # Display Gauge Chart
        st.plotly_chart(fig)

    except Exception as e:
        # Fallback to simple prediction if `predict_proba()` fails
        prediction = model.predict(df)[0]
        st.subheader('📌 Prediction Result')
        result_text = '❌ **Churn (Yes)**' if prediction == 1 else '✅ **Loyal Customer (No)**'
        st.markdown(f"<h2 style='text-align: center; color: {'red' if prediction == 1 else 'green'}'>{result_text}</h2>", unsafe_allow_html=True)
