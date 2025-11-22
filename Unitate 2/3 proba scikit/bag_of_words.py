from sklearn.feature_extraction.text import CountVectorizer

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

# Create and fit the vectorizer
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(corpus2)

# Show feature names (vocabulary)
print("Vocabulary:", bow_vectorizer.get_feature_names_out())

# Convert sparse matrix to array for readability
print("\nBoW Matrix (Document-Term Matrix):\n", X_bow.toarray())