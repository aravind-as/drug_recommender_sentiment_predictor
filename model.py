from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

def train_and_evaluate_model(data):
    # Split data for ML model
    df_train, df_test = train_test_split(data, test_size=0.25, random_state=0)

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(df_train['cleaned_review'])
    y_train = df_train['sentiment']
    X_test = vectorizer.transform(df_test['cleaned_review'])
    y_test = df_test['sentiment']

    # Train the Random Forest classifier
    model = LinearSVC() 
    model.fit(X_train, y_train)

    # Predict the sentiment labels for the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy

def predict_sentiment(model, vectorizer, review):
    review_vectorized = vectorizer.transform([clean_review(review)])
    sentiment_prediction = model.predict(review_vectorized)
    sentiment_mapping = {2: "Positive", 1: "Neutral", 0: "Negative"}
    predicted_sentiment = sentiment_mapping[sentiment_prediction[0]]
    
    return predicted_sentiment
