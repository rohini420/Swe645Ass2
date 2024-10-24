import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Load the data into DataFrame
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Remove #EOF and strip the text
            cleaned_line = line.replace('#EOF', '').strip()
            if cleaned_line:  # Ignore empty lines
                data.append(cleaned_line)
    return data

# Load train and test data
train_file = '/Users/hemanthsairambhatlapenumarthi/Documents/jeena/test_data.txt'
test_file = '/Users/hemanthsairambhatlapenumarthi/Documents/jeena/train_data.txts'

train_data = load_data(train_file)
test_data = load_data(test_file)

# Ensure proper split of sentiment and review
def split_sentiment_review(data):
    sentiment = []
    reviews = []
    for line in data:
        try:
            # Split the first word as sentiment and the rest as the review
            sent, review = line.split(' ', 1)
            sentiment.append(int(sent))  # Convert to integer
            reviews.append(review)
        except ValueError:
            print(f"Skipping invalid line: {line}")  # Handle any improperly formatted lines
    return sentiment, reviews

# Split the train data into sentiment and reviews
sentiments, reviews = split_sentiment_review(train_data)

# Create DataFrame for train data
train_df = pd.DataFrame({
    'Sentiment': sentiments,
    'Review': reviews
})

# Create DataFrame for test data
test_df = pd.DataFrame(test_data, columns=['Review'])

# 2. Pre-process the Data
def preprocess_text(text):
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize and remove stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Join the words back into a string
    return ' '.join(words)

# Apply pre-processing
train_df['Cleaned_Review'] = train_df['Review'].apply(preprocess_text)
test_df['Cleaned_Review'] = test_df['Review'].apply(preprocess_text)

# 3. Feature Engineering (TF-IDF Vectorizer)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))

# Fit on training data and transform both train and test data
X_train = vectorizer.fit_transform(train_df['Cleaned_Review'])
X_test = vectorizer.transform(test_df['Cleaned_Review'])

# Convert sentiment to integers (+1 or -1)
y_train = train_df['Sentiment']

# 4. Implement k-NN Classifier
knn_model = KNeighborsClassifier(n_neighbors=5, metric='cosine', weights = 'weights')

# 5. Implement Cross-Validation (Hyperparameter Tuning)
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11,15],
    'metric': ['cosine', 'euclidean', 'manhattan']
}

#grid_search = GridSearchCV(knn_model, param_grid, cv=3, scoring='accuracy')
#grid_search.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=200)

grid_search = GridSearchCV(log_reg, param_grid={'C': [0.01, 0.1, 1, 10]}, cv=3 ,scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and model
best_knn = grid_search.best_estimator_
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best cross-validation accuracy: {grid_search.best_score_}')


# 6. Predict on Test Data
test_predictions = best_knn.predict(X_test)

# Save the predictions to a file (for submission)
output_df = pd.DataFrame(test_predictions, columns=['Sentiment'])
output_df.to_csv("/Users/hemanthsairambhatlapenumarthi/Documents/jeena/format.txt", index=False, header=False)