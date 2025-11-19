import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

selected_features = ['Airbags', 'Prod. year', 'Mileage', 'Drive wheels', 'Wheel',
                     'Gear box type', 'Manufacturer', 'Model', 'Levy', 'Fuel type',
                     'Leather interior', 'Category', 'Cylinders']
def train_linear_regression(train_cleaned, test_cleaned, selected_features, target_column='Price'):
    print("run started")
    X = train_cleaned[selected_features]
    y = train_cleaned[target_column].astype(int)

    X2 = test_cleaned[selected_features]
    y2 = test_cleaned[target_column].astype(int)

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X2_scaled = scaler.transform(X2)

    linreg_model = LinearRegression()
    linreg_model.fit(X_train_scaled, y_train)

    y_train_pred = linreg_model.predict(X_train_scaled)
    y_validation_pred = linreg_model.predict(X_validation_scaled)
    y2_pred = linreg_model.predict(X2_scaled)

    metrics = {
        "Training R2": r2_score(y_train, y_train_pred),
        "Validation R2": r2_score(y_validation, y_validation_pred),
        "Validation MSE": mean_squared_error(y_validation, y_validation_pred),
        "Validation MAE": mean_absolute_error(y_validation, y_validation_pred)
    }

    return linreg_model, scaler, metrics, y2_pred
def train_svr_model(train_cleaned, test_cleaned, selected_features, target_column='Price'):
    print("run started")

    X = train_cleaned[selected_features]
    y = train_cleaned[target_column].astype(int)

    X2 = test_cleaned[selected_features]
    y2 = test_cleaned[target_column].astype(int)

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X2_scaled = scaler.transform(X2)

    svr_model = SVR(kernel='rbf', C=300, epsilon=0.05)
    svr_model.fit(X_train_scaled, y_train)

    y_train_pred = svr_model.predict(X_train_scaled)
    y_validation_pred = svr_model.predict(X_validation_scaled)
    y2_pred = svr_model.predict(X2_scaled)

    metrics = {
        "Training R2": r2_score(y_train, y_train_pred),
        "Validation R2": r2_score(y_validation, y_validation_pred),
        "Validation MSE": mean_squared_error(y_validation, y_validation_pred),
        "Validation MAE": mean_absolute_error(y_validation, y_validation_pred),
        "Final Test R2": r2_score(y2, y2_pred),
        "Final Test MSE": mean_squared_error(y2, y2_pred),
        "Final Test MAE": mean_absolute_error(y2, y2_pred),
        "Validation MAE (% of mean price)": mean_absolute_error(y_validation, y_validation_pred) / y_validation.mean() * 100,
        "Final Test MAE (% of mean price)": mean_absolute_error(y2, y2_pred) / y2.mean() * 100
    }

    return svr_model, scaler, metrics, y2_pred
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import randint

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import randint

def train_and_evaluate_random_forest(train_df, test_df, selected_features, target_column='Price'):
    X = train_df[selected_features]
    y = train_df[target_column].astype(int)
    X2 = test_df[selected_features]
    y2 = test_df[target_column].astype(int)

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_scaler = StandardScaler()
    X_train_scaled = rf_scaler.fit_transform(X_train)
    X_validation_scaled = rf_scaler.transform(X_validation)
    X2_scaled = rf_scaler.transform(X2)

    param_dist = {
        'n_estimators': randint(150, 500),
        'max_depth': randint(10, 30),
        'min_samples_split': randint(2, 15),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True]
    }

    rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train_scaled, y_train)
    best_rf = random_search.best_estimator_

    best_rf.fit(X_train_scaled, y_train)

    y_train_pred = best_rf.predict(X_train_scaled)
    y_val_pred = best_rf.predict(X_validation_scaled)
    y_test_pred = best_rf.predict(X2_scaled)

    metrics = {
        "Best Parameters": random_search.best_params_,
        "Train R2": r2_score(y_train, y_train_pred),
        "Train MSE": mean_squared_error(y_train, y_train_pred),
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Train MAE %": mean_absolute_error(y_train, y_train_pred) / y_train.mean() * 100,

        "Validation R2": r2_score(y_validation, y_val_pred),
        "Validation MSE": mean_squared_error(y_validation, y_val_pred),
        "Validation MAE": mean_absolute_error(y_validation, y_val_pred),
        "Validation MAE %": mean_absolute_error(y_validation, y_val_pred) / y_validation.mean() * 100,

        "Test R2": r2_score(y2, y_test_pred),
        "Test MSE": mean_squared_error(y2, y_test_pred),
        "Test MAE": mean_absolute_error(y2, y_test_pred),
        "Test MAE %": mean_absolute_error(y2, y_test_pred) / y2.mean() * 100
    }

    return best_rf, metrics, rf_scaler



def train_neural_network(train_cleaned, test_cleaned, selected_features, target_column='Price'):
    X = train_cleaned[selected_features]
    y = train_cleaned[target_column].astype(int)

    X2 = test_cleaned[selected_features]
    y2 = test_cleaned[target_column].astype(int)

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X2_scaled = scaler.transform(X2)

    mlp_model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    mlp_model.fit(X_train_scaled, y_train)

    y_train_pred = mlp_model.predict(X_train_scaled)
    y_validation_pred = mlp_model.predict(X_validation_scaled)
    y2_pred = mlp_model.predict(X2_scaled)

    def evaluate(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rel_mae = mae / y_true.mean() * 100
        return {'r2': r2, 'mse': mse, 'mae': mae, 'rel_mae_pct': rel_mae}

    metrics = {
        'train': evaluate(y_train, y_train_pred),
        'validation': evaluate(y_validation, y_validation_pred),
        'test': evaluate(y2, y2_pred)
    }


    return mlp_model, scaler, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def train_xgboost_model(train_cleaned, test_cleaned, selected_features, target_column='Price'):
    X = train_cleaned[selected_features]
    y = train_cleaned[target_column].astype(int)

    X2 = test_cleaned[selected_features]
    y2 = test_cleaned[target_column].astype(int)

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X2_scaled = scaler.transform(X2)

    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train_scaled, y_train)

    y_train_pred = xgb_model.predict(X_train_scaled)
    y_validation_pred = xgb_model.predict(X_validation_scaled)
    y2_pred = xgb_model.predict(X2_scaled)

    def evaluate(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rel_mae = mae / y_true.mean() * 100
        return {'r2': r2, 'mse': mse, 'mae': mae, 'rel_mae_pct': rel_mae}

    metrics = {
        'train': evaluate(y_train, y_train_pred),
        'validation': evaluate(y_validation, y_validation_pred),
        'test': evaluate(y2, y2_pred)
    }

    print(" XGBoost Regression Results\n")
    for split in ['train', 'validation', 'test']:
        print(f"{split.capitalize()} Metrics:")
        for metric_name, value in metrics[split].items():
            if metric_name == 'rel_mae_pct':
                print(f"  {metric_name}: {value:.2f}%")
            else:
                print(f"  {metric_name}: {value}")
        print()

    return xgb_model, scaler, metrics



if __name__ == "__main__":
    train_cleaned = pd.read_csv('train_cleaned_new_1.csv')
    test_cleaned = pd.read_csv('test_cleaned_new_1.csv')
    print("started neural training!")

    # model, scaler, metrics, _ = train_linear_regression(train_cleaned, test_cleaned, selected_features)

    # Save model and scaler
    #joblib.dump(model, 'linear_regression_model_new_1.pkl')
    #joblib.dump(scaler, 'scaler_l.pkl')

    #print("Training complete! Model and scaler saved.")
    # print(metrics)
    #svr_model, svr_scaler, svr_metrics, _ = train_svr_model(train_cleaned, test_cleaned, selected_features)
    #joblib.dump(svr_model, 'svr_model.pkl')
    #joblib.dump(svr_scaler, 'svr_scaler.pkl')
    #joblib.dump(svr_metrics, 'svr_metrics.pkl')
    """"best_rf_model, rf_metrics, rf_scaler = train_and_evaluate_random_forest(train_cleaned, test_cleaned, selected_features)
    joblib.dump(best_rf_model, 'rf_model.pkl')
    joblib.dump(rf_scaler, 'rf_scaler.pkl')
    joblib.dump(rf_metrics, 'rf_metrics.pkl')"""
    # xgb_model, xgb_scaler, xgb_metrics = train_xgboost_model(train_cleaned, test_cleaned, selected_features)
    # joblib.dump(xgb_model, 'xgb_model.pkl')
    # joblib.dump(xgb_scaler, 'xgb_scaler.pkl')
    # joblib.dump(xgb_metrics, 'xgb_metrics.pkl')
    neural_model, neural_scaler, neural_metrics = train_neural_network(train_cleaned, test_cleaned, selected_features)
    joblib.dump(neural_model, 'neural_model.pkl')
    joblib.dump(neural_scaler, 'neural_scaler.pkl')
    joblib.dump(neural_metrics, 'neural_metrics.pkl')

    



