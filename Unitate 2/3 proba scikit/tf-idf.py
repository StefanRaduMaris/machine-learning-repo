from sklearn.feature_extraction.text import TfidfVectorizer

# Mini dataset of sample reviews
corpus = [
    "I love this product",
    "This product is not good",
    "Absolutely fantastic experience",
    "Terrible, I hate it",
    "Not great, not terrible"
]

corpus2 = [
    "Iubesc acest produs",
    "Curierul mi-a pierdut coletul",
    "O experienta placuta",
    "Urasc cand ma pun sa merg dupa ei la sediu",
    "Asa si asa pana la urma decent "
]

# Create and fit the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus2)

# Show feature names
print("Vocabulary:", tfidf_vectorizer.get_feature_names_out())

# Convert to array and print
print("\nTF-IDF Matrix:\n", X_tfidf.toarray())