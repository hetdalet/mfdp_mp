from catboost import CatBoostRegressor


params = {
    "grow_policy": "Lossguide",
    "loss_function": "MAE",
    "iterations": 1700,
    "depth": 8,
    "min_data_in_leaf": 10,
    "learning_rate": 0.07688083438852199
}
model = CatBoostRegressor(**params)
