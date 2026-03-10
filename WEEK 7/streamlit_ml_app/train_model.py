import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# sample dataset
data = {
    "experience":[1,2,3,4,5,6,7,8,9,10],
    "salary":[30000,35000,40000,50000,60000,70000,80000,90000,100000,110000]
}

df = pd.DataFrame(data)

X = df[["experience"]]
y = df["salary"]

model = LinearRegression()
model.fit(X,y)

# save model
pickle.dump(model, open("salary_model.pkl","wb"))

print("Model trained and saved")