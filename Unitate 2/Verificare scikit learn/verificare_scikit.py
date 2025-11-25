
from sklearn import datasets
from pprint import pprint
import pandas as pd

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

print(df.dtypes)
df.to_csv("iris.csv")

