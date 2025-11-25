import pandas as pd 
#citim fisierul
df = pd.read_csv("IMLP4_01-reviews_unlabeled.csv")
#analizam datasetul
print(f"Forma {df.shape}")
#verificam structura
print(f"Coloanele sunt {df.columns}")

#Verificam daca avem valori lipsa
print(df.isna().sum())


#din acest dataset vom lua un sample
sample = print(df['review'].sample(10,random_state=33))

df["lungime_review"]=df["review"].astype(str).apply(lambda x : len(x.split()))

print(df["lungime_review"])

# lungimea medie a review-ului
lungime_medie=df["lungime_review"].mean()

print(f'Lungimea medie a raspunsurilor este {round(lungime_medie)}')


