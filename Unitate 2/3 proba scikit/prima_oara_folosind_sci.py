import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the labeled dataset
df = pd.read_csv("clean_data.csv")

# Separate features and labels
X = df['review']
y = df['sentiment']

# Vectorize all input reviews
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Train the logistic regression model on the full dataset
model = LogisticRegression()
model.fit(X_tfidf, y)

print("\nModel is ready.")

# Loop for user input
print("\nModel is ready. Enter a review to classify its sentiment.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter a review: ")
    if user_input.lower() == 'exit':
        print("Exiting sentiment classifier.")
        break

    # Transform user input to match TF-IDF vector format
    user_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(user_tfidf)[0]

    print(f"Predicted sentiment: {prediction}\n")