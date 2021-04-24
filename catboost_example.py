import pickle
from catboost import CatBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_training():
  trip_data = pickle.load(open( "save.p", "rb" ))
  models = list()
  if (len(trip_data) > 5):
    mlb = MultiLabelBinarizer()
    y_raw = trip_data["tag_array"]
    mlb.fit(y_raw)
    y = mlb.transform(y_raw)
    X = trip_data[['distance', 'start_long', 'start_lat', 'end_long', 'end_lat', 'start_hour', 'end_hour', 'vehicleid', 'sample_weight', 'vehicle_engine_capacity', 'vehicle_year']]

    num_tagged_trips = len(y[y])
    # split data into train and test sets
    seed = 7
    test_size = 0.33

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    print("Y_train")
    print(y_train)
    estimator = CatBoostClassifier(iterations=10,random_state=1, allow_const_label=True)
    model = OneVsRestClassifier(estimator=estimator)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_transformed = mlb.inverse_transform(y_pred)
    y_test_transformed = mlb.inverse_transform(y_test)
    print("y_pred")
    print(y_pred_transformed)
    print("y_test")
    print(y_test_transformed)
    predictions = [(value) for value in y_pred]
    print("predictions")
    print(predictions)
    accuracy = accuracy_score(y_test, predictions)
    print(f"accuracy {accuracy}")

    ACCURACY_THRESHOLD = 0.85
    if (accuracy > ACCURACY_THRESHOLD):
      models.append({
        "model": model,
        "tag_id": tag_id,
        "accuracy": accuracy
      })

  return models

test_training()