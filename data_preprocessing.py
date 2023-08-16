import re
import html
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

nltk.download('stopwords')
stop = set(stopwords.words('english'))

def load_data(train_data, test_data):
    train_data = pd.read_csv('C:/Users/91952/Documents/DRSAPP/data/drugsComTrain_raw.tsv', sep='\t')
    test_data = pd.read_csv('C:/Users/91952/Documents/DRSAPP/data/drugsComTest_raw.tsv', sep='\t')
    combined_data = pd.concat([train_data, test_data], ignore_index=True)

    # Drop rows with missing values
    filtered_df = combined_data.dropna()

    # Drop rows with usefulCount = 0
    filtered_df = filtered_df[filtered_df['usefulCount'] != 0]

    return filtered_df

def clean_review(review):
    # Remove HTML tags
    review = re.sub(r'<.*?>', '', str(review))

    # Unescape HTML entities
    review = html.unescape(review)

    # Removing Stopwords
    words = review.split()
    review = [w for w in words if not w in stop]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review]

    # Join words with space
    return ' '.join(review)


def preprocess_data(data):
    # Adding sentiment polarity column
    data['sentiment_polarity'] = data['review'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Labeling sentiment based on your conditions
    def label_sentiment(row):
        if row['rating'] > 7 and row['sentiment_polarity'] > 0.065:
            return 2
        elif 4 < row['rating'] <= 7 and row['sentiment_polarity'] > 0:
            return 1
        elif row['rating'] > 7 and row['sentiment_polarity'] < 0.065:
            return 1
        else:
            return 0

    data['sentiment'] = data.apply(label_sentiment, axis=1)

    # Balancing the dataset
    min_sample_count = min(len(data[data['sentiment'] == 2]),
                           len(data[data['sentiment'] == 1]),
                           len(data[data['sentiment'] == 0]))

    sentiment_2 = data[data['sentiment'] == 2].sample(n=min_sample_count, random_state=42)
    sentiment_1 = data[data['sentiment'] == 1].sample(n=min_sample_count, random_state=42)
    sentiment_0 = data[data['sentiment'] == 0].sample(n=min_sample_count, random_state=42)

    balanced_df = pd.concat([sentiment_2, sentiment_1, sentiment_0])
    
    return balanced_df
