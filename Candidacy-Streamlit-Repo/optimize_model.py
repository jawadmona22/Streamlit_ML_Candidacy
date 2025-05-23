import optuna
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from xgboost import XGBRegressor




def create_nn_model(trial):
    # Define the Keras model
    model = tf.keras.Sequential()
    hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(32,), (64,), (32, 32), (64, 32)])
    for size in hidden_layer_sizes:
        model.add(tf.keras.layers.Dense(size, activation='relu'))
    model.add(tf.keras.layers.Dense(1))  # Output layer

    # Learning rate
    learning_rate = trial.suggest_loguniform("learning_rate_init", 1e-5, 1e-1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    return model

def objective(trial, X_train, X_test, y_train, y_test):
    # Suggest the model type to use
    model_type = trial.suggest_categorical("model_type", [
        "random_forest", "gradient_boosting", "ridge", "lasso", "elastic_net",
        "svr", "knn", "neural_network", "xgboost"
    ])
    alpha = trial.suggest_float("alpha", 1e-3, 10.0, log=True)

    # Define the model and parameters based on the selected type
    if model_type == "random_forest":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

    elif model_type == "gradient_boosting":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

    elif model_type == "ridge":
        model = Ridge(alpha=alpha)

    elif model_type == "lasso":
        model = Lasso(alpha=alpha)

    elif model_type == "elastic_net":
        l1_ratio = trial.suggest_uniform("l1_ratio", 0, 1)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    elif model_type == "svr":
        C = trial.suggest_float("C", 1e-2, 100.0, log=True)
        epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
        model = SVR(C=C, epsilon=epsilon)

    elif model_type == "knn":
        n_neighbors = trial.suggest_int("n_neighbors", 3, 20)
        model = KNeighborsRegressor(n_neighbors=n_neighbors)

    elif model_type == "neural_network":
        model = create_nn_model(trial)

    elif model_type == "xgboost":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, use_label_encoder=False, eval_metric='rmse')

    # Fit the model and make predictions
    if model_type == "neural_network":
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0, validation_split=0.2)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    return mse

def optimize_model(features,target):

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train_scaled, X_test_scaled, y_train, y_test), n_trials=50)
    best_trial = study.best_trial
    print("Best trial parameters:", best_trial.params)
    print("Best trial MSE:", best_trial.value)

    return study


def RebuildBestModel(study):
    best_params = study.best_params
    model_type = best_params["model_type"]

    if model_type == "random_forest":
        # Rebuild the Random Forest model
        best_model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"]
        )

    elif model_type == "gradient_boosting":
        # Rebuild the Gradient Boosting model
        best_model = GradientBoostingRegressor(
            n_estimators=best_params["n_estimators"],
            learning_rate=best_params["learning_rate"],
            max_depth=best_params["max_depth"]
        )

    elif model_type == "ridge":
        # Rebuild the Ridge model
        best_model = Ridge(alpha=best_params["alpha"])

    elif model_type == "lasso":
        # Rebuild the Lasso model
        best_model = Lasso(alpha=best_params["alpha"])

    elif model_type == "elastic_net":
        # Rebuild the ElasticNet model
        best_model = ElasticNet(
            alpha=best_params["alpha"],
            l1_ratio=best_params["l1_ratio"]
        )

    elif model_type == "svr":
        # Rebuild the Support Vector Regression (SVR) model
        best_model = SVR(
            C=best_params["C"],
            epsilon=best_params["epsilon"]
        )

    elif model_type == "knn":
        # Rebuild the K-Nearest Neighbors model
        best_model = KNeighborsRegressor(n_neighbors=best_params["n_neighbors"])

    elif model_type == "neural_network":
        # Rebuild the Neural Network model (MLPRegressor)
        best_model = MLPRegressor(
            hidden_layer_sizes=best_params["hidden_layer_sizes"],
            alpha=best_params["alpha"],
            learning_rate_init=best_params["learning_rate_init"],
            max_iter=500
        )

    elif model_type == "xgboost":
        n_estimators = best_params["n_estimators"]
        max_depth = best_params["max_depth"]
        learning_rate = best_params["learning_rate"]
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, use_label_encoder=False, eval_metric='rmse')


    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return best_model