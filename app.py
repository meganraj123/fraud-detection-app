import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import load_model

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import time
import warnings
from sklearn.exceptions import InconsistentVersionWarning

import warnings
from sklearn.exceptions import InconsistentVersionWarning

import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import os

# Set your AWS credentials as environment variables for security
os.environ["AWS_ACCESS_KEY_ID"] = "AKIAWN26JUG5C2VREMFE"
os.environ["AWS_SECRET_ACCESS_KEY"] = "b7BLj88OMyVeYUN0XIzJ4GT3Pf03RR5a8C8e3oGe"
AWS_REGION = "ap-southeast-2"
SENDER_EMAIL = "meganraj020520@gmail.com"  # Replace with SES-verified sender email
RECIPIENT_EMAIL = "meganraj020520@gmail.com"  # Replace with SES-verified recipient email

# Initialize the SES client
ses_client = boto3.client('ses', region_name=AWS_REGION)

# Function to send email via Amazon SES
def send_email_via_ses(subject, body_text, recipient, sender="meganraj020520@gmail.com"):
    # Initialize the boto3 client for SES
    ses_client = boto3.client('ses', region_name="ap-southeast-2")

    # Construct the email
    try:
        response = ses_client.send_email(
            Destination={
                'ToAddresses': [recipient],  # List of recipients
            },
            Message={
                'Body': {
                    'Text': {
                        'Charset': "UTF-8",
                        'Data': body_text,
                    },
                },
                'Subject': {
                    'Charset': "UTF-8",
                    'Data': subject,
                },
            },
            Source=sender,
        )
        print(f"Email sent! Message ID: {response['MessageId']}")
    except NoCredentialsError:
        print("Error: AWS credentials not found.")
    except ClientError as e:
        print(f"Error: {e.response['Error']['Message']}")

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Enhanced Sidebar Navigation with icons
st.sidebar.title("Navigation")
st.sidebar.markdown(
    """
    <style>
    .nav-link {
        font-size: 18px;
        font-weight: 600;
        color: #007BFF;
        margin-bottom: 10px;
        padding: 10px 15px;
        border-radius: 10px;
        background-color: #F5F5F5;
    }
    .nav-link:hover {
        background-color: #007BFF;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True
)

page = st.sidebar.radio("Go to", [
    "🏠 Fraud Detection",
    "📊 Statistics",
    "📈 Analysis",
    "🔍 Fraud Prevention Monitoring",
    "👵 Elderly People Protection"
])

# Fraud Detection Logistic Regression Page
if page == "🏠 Fraud Detection":
    st.title("🏠 Fraud Detection Logistic Regression")

    st.write("""
    This is a fraud detection dashboard using a Logistic Regression model. Please input values for the following features to predict the likelihood of fraud.
    """)

    # Load the pre-trained Logistic Regression model
    model = joblib.load('logistic_regression.pkl')

    # Load preprocessor (for scaling features)
    preprocessor = joblib.load('preprocessor.pkl')

    # Sidebar for user input features
    st.sidebar.header("Input Features")


    def user_input_features():
        velocity_24h = st.sidebar.slider('Velocity 24h', 0.0, 10000.0, 5000.0)
        income = st.sidebar.slider('Income', 0.0, 1.0, 0.5)
        credit_risk_score = st.sidebar.slider('Credit Risk Score', -500, 500, 0)
        email_is_free = st.sidebar.selectbox('Is Email Free?', [0, 1])
        has_other_cards = st.sidebar.selectbox('Has Other Cards?', [0, 1])
        customer_age = st.sidebar.slider('Customer Age', 18, 100, 40)
        velocity_6h = st.sidebar.slider('Velocity 6h', 0.0, 20000.0, 5000.0)
        velocity_4w = st.sidebar.slider('Velocity 4w', 0.0, 10000.0, 5000.0)
        zip_count_4w = st.sidebar.slider('Zip Count 4w', 0, 10000, 50)

        # Store input values in a dictionary
        data = {
            'velocity_24h': velocity_24h,
            'income': income,
            'credit_risk_score': credit_risk_score,
            'email_is_free': email_is_free,
            'has_other_cards': has_other_cards,
            'customer_age': customer_age,
            'velocity_6h': velocity_6h,
            'velocity_4w': velocity_4w,
            'zip_count_4w': zip_count_4w
        }
        return pd.DataFrame(data, index=[0])


    # Input features as a DataFrame
    input_df = user_input_features()

    # Display the input features
    st.subheader("User Input Features")
    st.write(input_df)

    # Preprocess the input (scaling the input values)
    scaled_input = preprocessor.transform(input_df)

    # Make predictions using the neural network model
    predictions = model.predict(scaled_input)

    # Handle prediction output (assuming binary classification)
    if len(predictions.shape) == 1:
        prediction_prob = predictions[0]
    else:
        prediction_prob = predictions[0][0]  # Assuming binary classification with fraud probability

    # Display the prediction
    st.subheader("Prediction Probability")
    st.write(f"Fraud Probability: {prediction_prob:.2f}")

    # Decision based on threshold (e.g., 0.5)
    threshold = 0.5
    if prediction_prob >= threshold:
        st.error("This transaction is likely fraudulent.")
    else:
        st.success("This transaction is unlikely to be fraudulent.")

# Statistics Page
elif page == "📊 Statistics":
    st.title("📊 Statistics Page")

    # Load the preprocessed data
    data = pd.read_excel('preprocessed_data.xlsx')

    # 1. Summary Statistics
    st.header("Summary Statistics")
    st.write("Here are the key statistics for each feature in the dataset:")

    # Display the summary statistics
    st.write(data.describe())

    # 2. Visualizations
    st.header("Feature Distributions")

    # Distribution plots for each feature
    for column in data.columns[:-1]:  # Exclude the target variable 'fraud_bool'
        st.subheader(f"Distribution of {column}")

        fig, ax = plt.subplots()
        sns.histplot(data[column], kde=True, ax=ax, color="skyblue", bins=30)
        st.pyplot(fig)

    # 3. Fraud Count (Fraud vs Non-Fraud)
    st.header("Fraud vs Non-Fraud Transactions")

    fraud_counts = data['fraud_bool'].value_counts()

    fig, ax = plt.subplots()
    sns.barplot(x=fraud_counts.index, y=fraud_counts.values, ax=ax, palette="viridis")
    ax.set_title('Count of Fraudulent (1) vs Non-Fraudulent (0) Transactions')
    ax.set_xlabel('Fraud Indicator (fraud_bool)')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # 4. Correlation Matrix
    st.header("Correlation Matrix")
    st.write("The following matrix shows the correlation between different features:")

    # Correlation matrix
    correlation_matrix = data.corr()

    # Plot the heatmap for the correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)

# Analysis Page
elif page == "📈 Analysis":
    st.title("📈 Analysis Page: Feature Importance and Comparisons")

    st.write("""
    On this page, you can explore deeper insights into feature importance and pairwise feature comparisons.
    """)

    # Load the preprocessed data
    data = pd.read_excel('preprocessed_data.xlsx')

    # Load the pre-trained Neural Network model
    model = joblib.load('logistic_regression.pkl')

    # Load preprocessor (for scaling features)
    preprocessor = joblib.load('preprocessor.pkl')

#################################################################################################################################################################

    # Feature Importance using SHAP
    st.header("Feature Importance using SHAP")

    # Select a sample of data to compute SHAP values
    shap_sample_data = data.sample(100)  # Limit to 100 samples to reduce computation time

    # Load the SHAP explainer for the Logistic Regression model
    explainer = shap.KernelExplainer(model.predict, preprocessor.transform(shap_sample_data.drop('fraud_bool', axis=1)))
    shap_values = explainer.shap_values(preprocessor.transform(shap_sample_data.drop('fraud_bool', axis=1)))

    # Plot the SHAP summary plot for feature importance
    st.subheader("SHAP Summary Plot")

    # Generate the SHAP summary plot
    fig_shap, ax_shap = plt.subplots()
    shap.summary_plot(shap_values, shap_sample_data.drop('fraud_bool', axis=1), plot_type="bar", show=False)
    st.pyplot(fig_shap)

###################################################################################################################################################################

    # Pairwise Comparison: Scatter plots between important features
    st.header("Pairwise Feature Comparison")

    selected_feature_x = st.selectbox("Select X-axis feature", data.columns)
    selected_feature_y = st.selectbox("Select Y-axis feature", data.columns)

    # Generate scatter plot for the selected features
    fig, ax = plt.subplots()
    sns.scatterplot(x=selected_feature_x, y=selected_feature_y, hue="fraud_bool", data=data, ax=ax)
    plt.title(f"Scatter Plot: {selected_feature_x} vs {selected_feature_y}")
    st.pyplot(fig)

    # Boxplot comparison: Feature vs Target variable (fraud_bool)
    st.header("Feature vs Fraud Comparison (Box Plot)")

    selected_boxplot_feature = st.selectbox("Select feature for box plot", data.columns[:-1])  # Exclude fraud_bool
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='fraud_bool', y=selected_boxplot_feature, data=data, ax=ax2)
    plt.title(f"Box Plot: {selected_boxplot_feature} vs Fraud")
    st.pyplot(fig2)

# Fraud Prevention Monitoring System Page
elif page == "🔍 Fraud Prevention Monitoring":
    st.title("🔍 Fraud Prevention Monitoring System")

    st.write("""
    This page provides a monitoring system to track and detect fraudulent transactions in real-time. It includes key fraud metrics and an alert system.
    """)

    # Simulating real-time fraud detection with sample data
    def generate_fake_data(size=1):
        """
        Generate a fake dataset of transactions with random values for features
        """
        data = {
            'Transaction_ID': np.arange(1, size + 1),
            'velocity_24h': np.random.uniform(0, 10000, size),
            'income': np.random.uniform(0, 1, size),
            'credit_risk_score': np.random.randint(-500, 500, size),
            'email_is_free': np.random.randint(0, 2, size),
            'has_other_cards': np.random.randint(0, 2, size),
            'customer_age': np.random.randint(18, 100, size),
            'velocity_6h': np.random.uniform(0, 20000, size),
            'velocity_4w': np.random.uniform(0, 10000, size),
            'zip_count_4w': np.random.randint(0, 10000, size),
            'fraud_bool': np.random.randint(0, 2, size)
        }
        return pd.DataFrame(data)

    # Simulate incoming transactions
    st.header("Real-Time Fraud Detection Monitoring")
    st.write("Monitoring real-time transactions for fraud...")

    # Initializing empty dataframe for monitoring
    monitored_data = pd.DataFrame()
    placeholder = st.empty()

    for _ in range(5):  # Simulating 5 rounds of new data
        # Simulating new batch of 10 transactions
        new_data = generate_fake_data(size=10)
        monitored_data = pd.concat([monitored_data, new_data], ignore_index=True)

        # Display the latest transactions in the placeholder
        with placeholder.container():
            st.subheader("Latest Transactions")
            st.dataframe(new_data)

            st.subheader("Current Fraud Detection Alerts")
            fraud_transactions = new_data[new_data['fraud_bool'] == 1]
            st.write(f"{len(fraud_transactions)} fraudulent transactions detected.")
            if not fraud_transactions.empty:
                st.write(fraud_transactions)

            # Plot fraud vs non-fraud detection over time
            st.subheader("Fraud Detection Over Time")
            fig, ax = plt.subplots()
            sns.lineplot(data=monitored_data['fraud_bool'].value_counts(normalize=True).sort_index(), ax=ax)
            ax.set_title("Fraud vs Non-Fraud Detection")
            st.pyplot(fig)

            # Pause to simulate real-time monitoring
            time.sleep(2)

    st.write("Monitoring complete.")

    # Displaying key metrics
    st.header("Key Fraud Metrics")

    total_transactions = len(monitored_data)
    total_fraud = monitored_data['fraud_bool'].sum()
    fraud_rate = (total_fraud / total_transactions) * 100

    st.metric("Total Transactions Monitored", total_transactions)
    st.metric("Total Fraudulent Transactions", total_fraud)
    st.metric("Fraud Detection Rate (%)", f"{fraud_rate:.2f}%")

    # Fraud Prevention Suggestions
    st.header("Fraud Prevention Suggestions")
    st.write("""
        Here are some suggested actions to improve fraud detection and prevention:

        1. **Implement two-factor authentication**: Ensure that high-risk transactions require two-factor authentication, especially for older or vulnerable users.
        2. **Regularly update machine learning models**: Periodically retrain models to improve detection rates based on the latest data.
        3. **Monitor anomalous behaviors**: Focus on transactions with unusual patterns like rapid spending or location changes.
        4. **Improve Customer Communication**: Send alerts to customers when suspicious transactions are detected.
        5. **Educate customers**: Help users understand the importance of secure passwords, and avoiding phishing scams.
        """)

    # Send Fraud Alert Notification through Amazon SES
    st.subheader("📧 Send Fraud Alert Notification")

    st.write("When fraud is detected, notify users via email or SMS.")
    email_alert = st.text_input("Enter your email to receive fraud alerts", "")

    if st.button("Send Fraud Alert"):
        if email_alert:
            # Example: You would send an alert here using Amazon SES or another service
            send_email_via_ses(subject="Fraud Alert", body_text="Fraud has been detected in your account",
                               recipient=email_alert)
            st.success(f"Fraud alert sent to {email_alert}!")
        else:
            st.error("Please enter a valid email address.")


# Elderly People Protection Page Simulation with UI Enhancements
elif page == "👵 Elderly People Protection":
    st.markdown("<h1 style='text-align: center; color: #FF6347;'>👵 Elderly People Protection</h1>",
                unsafe_allow_html=True)
    st.write("""
    This section provides fraud prevention simulations and tools specifically tailored for elderly users. The focus is on 
    Two-Factor Authentication (2FA), real-time alerts, and enhanced customer support.
    """)

    # Solution 1: Two-Factor Authentication (2FA) with Guardian Notification
    st.subheader("🔒 Two-Factor Authentication (2FA) with Guardian Notification")

    st.write("""
    Simulate a transaction where both the elderly user and the guardian receive an OTP to verify the transaction.
    """)

    # OTP simulation
    import random

    # Transaction simulation
    transaction_amount = st.number_input('Enter Transaction Amount', min_value=0.0, value=50.0, format="%.2f")
    user_phone = st.text_input('📱 Elderly User Phone Number', value='123-456-7890')
    guardian_phone = st.text_input('👤 Guardian Phone Number', value='987-654-3210')

    # Initialize session states
    if 'otp' not in st.session_state:
        st.session_state.otp = None
    if 'otp_sent' not in st.session_state:
        st.session_state.otp_sent = False
    if 'validation_status' not in st.session_state:
        st.session_state.validation_status = None

    # Sending OTP to user and guardian
    if st.button("Send OTP to User & Guardian"):
        st.session_state.otp = random.randint(100000, 999999)
        st.session_state.otp_sent = True
        st.session_state.validation_status = None
        st.info(f"Sending OTP to both {user_phone} and {guardian_phone}...")
        st.write(f"🛡️ OTP sent: {st.session_state.otp}")  # Simulate OTP sent

    # Check if OTP was sent before showing the input field
    if st.session_state.otp_sent:
        user_input_otp = st.text_input("Enter the OTP you received:")

        if st.button("Validate OTP"):
            # Check OTP
            if user_input_otp == str(st.session_state.otp):
                st.session_state.otp_validated = True
                if st.session_state.otp_validated is True:
                    st.success("🎉 Transaction Approved!")
                    st.balloons()  # Show success animation
                    # Disable the buttons after validation
                    st.session_state.otp_sent = False
                else:
                    st.error("❌ Invalid OTP! Transaction Denied.")
            else:
                st.session_state.otp_validated = False
                if st.session_state.otp_validated is False:
                    st.error("❌ Invalid OTP! Transaction Denied.")


    # Solution 2: 24/7 Dedicated Customer Support for Elderly Users
    st.subheader("📞 24/7 Dedicated Customer Support")

    st.write("""
    Elderly users can connect with our 24/7 support team to report issues or receive help with their transactions.
    """)

    # Simulate connecting to support
    if st.button("Contact Customer Support"):
        st.info("🔄 Connecting you to customer support...")
        st.success("✔️ You are now connected to a support agent!")

    # Solution 3: Real-Time Transaction Monitoring and Alerts
    st.subheader("📊 Real-Time Transaction Monitoring and Alerts")

    st.write("""
    This section demonstrates a transaction monitoring system that sends real-time alerts to the user and their guardian when suspicious activity is detected.
    """)

    # Monitoring threshold
    suspicious_threshold = 5000.0
    monitored_transaction_amount = st.number_input('Enter Transaction Amount for Monitoring', min_value=0.0, value=50.0,
                                                   format="%.2f")

    if st.button("Monitor Transaction"):
        if monitored_transaction_amount > suspicious_threshold:
            st.error(f"🚨 Suspicious transaction detected for ${monitored_transaction_amount:.2f}!")
            st.write(f"🔔 Notifying guardian at {guardian_phone}.")
        else:
            st.success(f"✔️ Transaction of RM{monitored_transaction_amount:.2f} seems normal.")

    # Solution 4: Simplified User Interface for Elderly Users
    st.subheader("🖱️ Simplified Transaction Interface")

    st.write("""
    We provide a simplified transaction interface with larger fonts and fewer steps, tailored for elderly users.
    """)

    st.markdown("<h2 style='font-size:30px;'>Simple Transaction</h2>", unsafe_allow_html=True)
    simple_transaction_amount = st.number_input('💵 Enter Amount', min_value=0.0, value=50.0, format="%.2f")

    if st.button("Complete Simple Transaction"):
        st.success(f"✔️ Transaction of RM{simple_transaction_amount:.2f} completed successfully!")

    # Solution 5: Biometric Authentication Simulation
    st.subheader("👁️ Biometric Authentication")

    st.write("""
    Use biometric authentication (e.g., fingerprint or facial recognition) to confirm elderly user identity.
    """)

    if st.button("Simulate Biometric Authentication"):
        st.info("🔄 Simulating biometric authentication...")
        st.success("✔️ Biometric authentication successful! Identity confirmed.")

    # Solution 6: Fraud Awareness and Education
    st.subheader("📚 Fraud Awareness and Education for Elderly Users")

    st.write("""
    We provide simple educational resources to help elderly users recognize and avoid common fraud tactics.
    """)

    st.markdown("""
    **Tips to Prevent Fraud:**
    - 🚫 Do not share passwords or personal information over the phone or email.
    - 🛡️ Verify the identity of the person requesting sensitive information.
    - 🔔 Set up transaction alerts for all banking activities.
    - 🔒 Keep your software and apps updated for security.
    """)

    st.info("For more information, visit our fraud prevention guide [here](https://www.bnm.gov.my/financial-fraud-alerts).")

