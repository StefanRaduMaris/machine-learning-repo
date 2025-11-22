import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("clean_data.csv")
X = df['review']
y = df['sentiment']

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

 

#Pasul 2: Antrenarea modelului
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_tfidf, y)

 

#Pasul 3: Testarea modelului
test_reviews = [
"Simple, fast, but feels cheap.",
"I hate how it looks but it performs well.",
"Absolutely amazing quality, I love it!",
"Worst product ever. Total disappointment."
]

test_tfidf = vectorizer.transform(test_reviews)

predictions = knn_model.predict(test_tfidf)

for review, sentiment in zip(test_reviews, predictions):
    print(f"{review} → {sentiment}")