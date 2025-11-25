#Importam bibliotecile necesare
#Vreau sa folosesc toti algoritmi pentru a determina daca un mesaj e spam sau nu
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

#citit csv-ul
df = pd.read_csv("messages.csv")
print(df['category'].value_counts())
print(f"Cate valori nu sunt etichetate : {df.isna().sum()}")
df['category']=df['category'].replace("not spam",'ham')

print(df['category'].value_counts())
df = df.dropna()
df.head(10)


#realizam cate o variabila in care vom pune datele din mesaj si datele din category
x=df['message']
y=df['category']


#facem transformarea lui x in vector
vectorize=TfidfVectorizer()
x_tf=vectorize.fit_transform(x)

#Realizam un dictionar care va avea cele 3 functii diferite
#Vom antrena fiecare model cu acelas timp de date
models = {
    "Logistic regression":LogisticRegression(max_iter=1000),
    "Multinomial":MultinomialNB(),
    "TreeClassifier":DecisionTreeClassifier()
}

#Vreau sa creez niste coloane noi in df pentru a vedea ce valori va lua fiecare algoritm in parte

trained_models = {}
for name, model in models.items():
    model.fit(x_tf, y)
    trained_models[name] = model
    df[name]=model.predict(x_tf)
print('Modelele sunt antrenate')


df['final'] = df[['Logistic regression', 'Multinomial', 'TreeClassifier']].mode(axis=1)[0]
df['final'] = df['final'].fillna(df['Logistic regression'])


import plotly.express as px
df["dimensiune"]=df["message"].apply(lambda x : len(str(x)))
print(df['dimensiune'])

fig=px.scatter(data_frame=df,
            x=df.index,
            y=df['dimensiune'],
            hover_name=df['final'],
            hover_data=('message','Logistic regression','Multinomial','TreeClassifier' ),
            color=df['final']
            )

fig.update_layout(

    xaxis_title="Lungime text",
    yaxis_title="Index",
    
)

fig.show()

while True:
    user_answer = input("Vrei sa testezi propriile exemple? (y/n): ").lower()
    if user_answer == "y":
        text_input = input("Introduceti mesajul: ")
        test_tf = vectorize.transform([text_input])
        predictie = trained_models["Logistic regression"].predict(test_tf)[0]
        print(f"Mesaj: {text_input} --> Predicție: {predictie.upper()}")
    else:
        break