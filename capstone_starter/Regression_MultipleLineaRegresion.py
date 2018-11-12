import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import timeit

df = pd.read_csv("profiles.csv")
body_type_df = df.dropna(subset=['body_type'])
print(df.body_type.value_counts())

drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4}
drugs = {"never": 0, "sometimes": 1, "often": 2}
diet = {"mostly anything" : 0, "anything" : 0, "strictly anything": 0,
        "mostly vegetarian": 1, "mostly other": 2, "strictly vegetarian": 1,
        "vegetarian": 1, "strictly other": 2, "other": 2, "mostly vegan": 3,
        "strictly vegan" : 3, "vegan": 3, "mostly kosher": 4, "mostly halal": 5,
        "strictly halal": 5, "strictly kosher": 4, "kosher": 4, "halal": 5}
body_type = {"average": 1, "fit": 2, "athletic": 3, "thin": 4, "curvy": 5,
"a little extra": 6, "skinny": 7, "full figured": 8, "overweight": 9, "jacked": 10,
"used up": 11, "rather not say": 12}

body_type_df["drinks_code"] = body_type_df.drinks.map(drink_mapping)
body_type_df["smokes_code"] = body_type_df.smokes.map(smokes_mapping)
body_type_df["drugs_code"] = body_type_df.drugs.map(drugs)
body_type_df["diet_code"] = body_type_df.diet.map(diet)
body_type_df["body_type"] = body_type_df.body_type.map(body_type)

feature_data = body_type_df[['drinks_code','smokes_code', 'drugs_code', 'age', 'diet_code', 'height']]
feature_data.fillna(value=0, inplace=True)

x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

body_type_labels = body_type_df["body_type"]
body_type_train, body_type_test, body_type_labels_train, body_type_labels_test = train_test_split(feature_data, body_type_labels, train_size=0.8, test_size=0.2,random_state=6)

### REGRESSION

print("\n### Multiple Linear Regression")

start = timeit.default_timer()

mlr = LinearRegression()

mlr.fit(body_type_train, body_type_labels_train)

prediction = mlr.predict(body_type_test)

stop = timeit.default_timer()

print("\nAccuracy score: " + str(accuracy_score(body_type_labels_test, prediction.round())))
print("Recall score: " + str(recall_score(body_type_labels_test, prediction.round(), average='micro')))
print("Precision score: " + str(precision_score(body_type_labels_test, prediction.round(), average='micro')))

print("\nTime to run the model: " + str(stop-start))