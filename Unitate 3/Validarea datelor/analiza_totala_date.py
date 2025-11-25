from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
df = pd.read_csv("clean_data.csv")
X = df["review"]
y = df["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(y_train.value_counts())
print(y_test.value_counts())

models={
    'LogisticRegression':LogisticRegression(max_iter=1000),
    'Bayes':MultinomialNB(),
    'Tree':DecisionTreeClassifier(),
    'LiniarSVC':LinearSVC()
}

for name,model in models.items():
    model.fit(X_train_vec,y_train)
    y_pred=model.predict(X_test_vec)
    print(f"\n{name} - Classification Report:")
    print(classification_report(y_test, y_pred))