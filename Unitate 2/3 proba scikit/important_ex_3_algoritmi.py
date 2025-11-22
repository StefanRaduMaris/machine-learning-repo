import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

# Load labeled dataset
df = pd.read_csv("clean_data.csv")
X = df['review']
y = df['sentiment']

# Convert text to TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Train different algorithms on the same data
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier()
}

trained_models = {}
for name, model in models.items():
    model.fit(X_tfidf, y)
    trained_models[name] = model

print("\nModels are trained. Type a review to classify it.")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("Enter a review: ")
    if user_input.lower() == 'exit':
        print("Exiting.")
        break

    user_tfidf = vectorizer.transform([user_input])

    for name, model in trained_models.items():
        prediction = model.predict(user_tfidf)[0]
        print(f"{name} → {prediction}")
    print("-" * 40)