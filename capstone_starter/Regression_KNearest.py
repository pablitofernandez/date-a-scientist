import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import timeit

df = pd.read_csv("profiles.csv")
body_type_df = df.dropna(subset=['body_type'])

drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4}
drugs = {"never": 0, "sometimes": 1, "often": 2}
diet = {"mostly anything" : 0, "anything" : 0, "strictly anything": 0,
        "mostly vegetarian": 1, "mostly other": 2, "strictly vegetarian": 1,
        "vegetarian": 1, "strictly other": 2, "other": 2, "mostly vegan": 3,
        "strictly vegan" : 3, "vegan": 3, "mostly kosher": 4, "mostly halal": 5,
        "strictly halal": 5, "strictly kosher": 4, "kosher": 4, "halal": 5}

body_type_df["drinks_code"] = body_type_df.drinks.map(drink_mapping)
body_type_df["smokes_code"] = body_type_df.smokes.map(smokes_mapping)
body_type_df["drugs_code"] = body_type_df.drugs.map(drugs)
body_type_df["diet_code"] = body_type_df.diet.map(diet)

feature_data = body_type_df[['drinks_code','smokes_code', 'drugs_code', 'age', 'diet_code', 'height']]
feature_data.fillna(value=0, inplace=True)

x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

body_type_labels = body_type_df["body_type"]
body_type_train, body_type_test, body_type_labels_train, body_type_labels_test = train_test_split(feature_data, body_type_labels, train_size=0.8, test_size=0.2,random_state=6)

### REGRESSION

print("\n### K-Nearest Neighbors Regression")

start_k = timeit.default_timer()

classifier = KNeighborsClassifier(n_neighbors=80)
classifier.fit(body_type_train, body_type_labels_train)

print("Score: " + str(classifier.score(body_type_test, body_type_labels_test)))

prediction = classifier.predict(body_type_test)

stop = timeit.default_timer()

print("Accuracy score: " + str(accuracy_score(body_type_labels_test, prediction)))
print("Recall score: " + str(recall_score(body_type_labels_test, prediction, average='micro')))
print("Precision score: " + str(precision_score(body_type_labels_test, prediction, average='micro')))

print("\nTime to run the model: " + str(stop - start_k))