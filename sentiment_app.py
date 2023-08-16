import streamlit as st
import data_preprocessing

from data_preprocessing import load_data, clean_review , preprocess_data
from model import train_and_evaluate_model, predict_sentiment  # Import the functions


def main():
    st.title("Drug Recommendation System")

    # Disclaimer
    st.write("Empowering Your Health Decisions: Expert Insights Await, Consultation Advised!")

    # Load data
    data = data_preprocessing.load_data('drugsComTrain_raw.tsv', 'drugsComTest_raw.tsv')
    # Preprocess the review column
    data['cleaned_review'] = data['review'].apply(clean_review)  
    data = data_preprocessing.preprocess_data(data)
    

    # Train and evaluate the ML model
    model, vectorizer, accuracy = train_and_evaluate_model(data)

    # Display the head of the combined dataset
    st.write("Sample of Dataset (Head 5 rows)")
    st.dataframe(data.head())

    st.sidebar.header("User Input")
    preference = st.sidebar.multiselect("Select your health condition:", data["condition"].unique())

    # Middle Section
    st.markdown("<h2 style='text-align: center; margin-top: 30px;'>Submit Review and Analyze Sentiment</h2>", unsafe_allow_html=True)
    review = st.text_area("Submit your review:")
    if st.button("Analyze Sentiment"):
        #sentiment = analyze_sentiment(review)  # Call your sentiment analysis function here
        predicted_sentiment = predict_sentiment(model, vectorizer, review)  # Use the ML model

        st.markdown("<h3 style='text-align: center; margin-top: 20px;'>Review Sentiment:</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 24px; margin-top: 10px;'>{sentiment} ({predicted_sentiment})</p>", unsafe_allow_html=True)

    # Recommendation
    st.sidebar.header("Recommendation")
    # Rest of your code...

if __name__ == "__main__":
    main()
