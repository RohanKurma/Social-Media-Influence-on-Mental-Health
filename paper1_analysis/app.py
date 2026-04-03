import streamlit as st
import pandas as pd
import joblib

# Load trained models
ucla_model = joblib.load("ucla_model.pkl")
phq_model = joblib.load("phq_model.pkl")
gad_model = joblib.load("gad_model.pkl")
# Load PSQI pipeline
psqi_model = joblib.load("psqi_model.pkl")

# Load pipeline for preprocessing
pipeline = joblib.load("ucla_pipeline.pkl")  # Assuming same preprocessing for all models

st.title("Social Media Mental Health Predictor")

st.markdown("### Please answer the following questions:")

# --- 1️⃣ User Input Form ---
with st.form("user_input_form"):
    q1 = st.selectbox("1. Do you have a social media account?", ["Yes", "No"])
    q2 = st.selectbox("2. Which social media account do you use usually?", ["Facebook", "Instagram", "Twitter", "Other"])
    q3 = st.selectbox("3. Which device do you usually use to connect social media?", ["Mobile", "Tablet", "Laptop/Desktop"])
    q4 = st.selectbox("4. Which type of internet connection do you use?", ["Broadband", "Mobile Data"])
    q5 = st.selectbox("5. How long have you been using social media?", ["less than 2-year", "2-5 years", "5-10 years", "more than 10 years"])
    q6 = st.selectbox("6. How frequently do you post on social media?", ["less than 1 per day", "1-2 per day", "3-5 per day", "more than 5 per day"])
    q7 = st.selectbox("7. How much time do you spend daily in social media?", ["less than 1 hour", "1-3 hours", "3-5 hours", "more than 5 hours"])
    q8 = st.selectbox("8. When do you usually use social media?", ["morning", "afternoon", "evening", "night"])
    q9 = st.selectbox("9. How many friends do you have on social media?", ["less than 500", "500-2000", "2000-4000", "more than 4000"])
    q10 = st.selectbox("10. How many friends do you know personally?", ["few of them", "many of them", "most of them", "all of them"])
    q11 = st.selectbox("11. How many groups are you tagged in?", ["less than 5", "5-10", "10-20", "more than 20"])
    q12 = st.selectbox("12. What is your main purpose for using social media?", ["Entertainment", "Networking", "News", "Business", "Education", "Other"])
    q13 = st.selectbox("13. What contents do you mainly look for in your social media news feed?", ["Memes", "News", "Educational", "Self-help", "Other"])
    q14 = st.selectbox("14. Do you believe social media is a good thing?", ["Yes", "No"])
    q15 = st.selectbox("15. When you see something in social media, do you instantly believe it?", ["Yes", "No"])
    q16 = st.selectbox("16. Have you ever experienced peer pressure due to social media?", ["Yes", "No"])
    q17 = st.selectbox("17. Does your emotion get influenced by others' posts?", ["not at all", "sometimes", "always"])
    q18 = st.selectbox("18. Have you compared yourself with others' success?", ["never", "most of the times", "all the times"])
    q19 = st.selectbox("19. Would your mental wellbeing be better without social media?", ["Yes", "No"])
    q20 = st.selectbox("20. Are you trying to reduce your use of social media?", [
        "no, i am not trying", "not trying", "trying to reduce the use",
        "yes, i am trying but can’t", "yes, i am trying and i have reduced using it", "trying to stop the use"
    ])

    submit = st.form_submit_button("Predict My Mental Health Scores")

# --- 2️⃣ Preprocess and Predict ---
if submit:
    # Build user input as DataFrame
    user_input = pd.DataFrame([[
        q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16, q17, q18, q19, q20
    ]], columns=[
        '1. Do you have a social media account? (e.g., Facebook, Twitter, etc.)',
        '2. Which social media account do you use usually?',
        '3. Which device do you usually use to connect social media?',
        '4. Which type of internet connection do you use?',
        '5. How long have you been using a social media account?',
        '6. How frequently do you post (upload status or add photos/videos) on social media? ',
        '7. How much time do you spend daily in social media?',
        '8. When do you usually use social media?',
        '9. How many friends do you have on social media?',
        '10. How many friends do you know personally in social media? ',
        '11. How many groups you are tagged in social media?',
        '12. What is your main purpose for using social media (e.g. Facebook)?',
        '13. What contents do you mainly look for in your social media news feed?',
        '14.Do you believe social media is a good thing?',
        '15. When you see something in social media, do you instantly believe it?',
        '16. Have you ever experienced peer pressure due to social media?',
        "17. Does your emotion get influenced by other's posts (success, failure, loss)? ",
        '18. Have you ever compared yourself with other’s success or luxurious life?',
        '19. Do you think, your mental wellbeing would be better if you do not use social media?',
        '20. If answer is yes, are you trying to control that thing and trying to reduce the use of social media?'
    ])

    # Preprocess user input
    processed_input = pipeline.transform(user_input)

    # Predict scores
    ucla_score = ucla_model.predict(processed_input)[0]
    phq_score = phq_model.predict(processed_input)[0]
    gad_score = gad_model.predict(processed_input)[0]
    psqi_score = psqi_model.predict(processed_input)[0]  # NEW: PSQI

    # --- 3️⃣ Show Results ---
    st.subheader("📋 Your Mental Health Scorecard")
    st.metric("UCLA Loneliness Score", f"{ucla_score:.2f}")
    st.metric("PHQ-9 Depression Score", f"{phq_score:.2f}")
    st.metric("GAD-7 Anxiety Score", f"{gad_score:.2f}")
    st.metric("PSQI Sleep Quality Score", f"{psqi_score:.2f}")  # NEW: Display PSQI
