import pandas as pd

#####Sa definim regulile ########
#In acest exemplu cuvintele din recenzii ne vor ghida pentru a crea regulile
#dacă recenzia conține „excelent”, „recomand”, „livrare rapidă” → pozitiv;
#dacă scrie „deteriorat”, „întârzie”, „dezamăgit” → negativ.

def clasificare_rugula(text):
    lista_pozitiv=["great", "excellent", "amazing", "love", "perfect", "good", "fantastic", "happy", "recommend"]
    lista_negativa=["bad", "terrible", "worst", "broken", "poor", "disappointed", "awful", "hate", "slow",'cheap']
    fraze_negative=['didn’t meet expectations', 'waste of money', 'heating issue',"cheap quality"]
    text=str(text).lower()

    if not any(t in text for t in fraze_negative):
        number_of_positive=sum(words in text for words in lista_pozitiv)
        number_of_negative=sum(words in text for words in lista_negativa)
        if number_of_positive >= number_of_negative:
            return "positive"
    else :
        return "negative"


df = pd.read_csv("clean_data.csv")
print(df.head(20))

df['clasificare']=df['review'].apply(clasificare_rugula)
print(df)

df.to_csv("check_data.csv")


