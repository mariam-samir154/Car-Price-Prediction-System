"""# linear regression model"""

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Select features and target
selected_features = ['Airbags', 'Prod. year', 'Mileage','Drive wheels','Wheel',
                     'Gear box type', 'Manufacturer','Model','Levy','Fuel type','Leather interior','Category','Cylinders']

X = train_cleaned[selected_features]
y = train_cleaned['Price'].astype(int)

X2 = test_cleaned[selected_features]
y2 = test_cleaned['Price'].astype(int)

# Split train_cleaned into training and validation sets
X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize (fit only on X_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X2_scaled = scaler.transform(X2)  # Final test set

# Train Linear Regression
linreg_model = LinearRegression()
linreg_model.fit(X_train_scaled, y_train)

# Predict
y_train_pred = linreg_model.predict(X_train_scaled)
y_validation_pred = linreg_model.predict(X_validation_scaled)
y2_pred = linreg_model.predict(X2_scaled)  # Final prediction on test_cleaned

# Evaluation Metrics
print("Training R² Score:", r2_score(y_train, y_train_pred))
print("Validation R² Score:", r2_score(y_validation, y_validation_pred))
print("Validation MSE:", mean_squared_error(y_validation, y_validation_pred))
print("Validation MAE:", mean_absolute_error(y_validation, y_validation_pred))

# Relative MAE for validation set
mean_val_price = y_validation.mean()
mae_val = mean_absolute_error(y_validation, y_validation_pred)
print(f"Validation MAE (% of mean price): {mae_val / mean_val_price * 100:.2f}%")

print("\n--- Final Test Set (test_cleaned) ---")
print("Final Test R² Score:", r2_score(y2, y2_pred))
print("Final Test MSE:", mean_squared_error(y2, y2_pred))
print("Final Test MAE:", mean_absolute_error(y2, y2_pred))

# Relative MAE for final test set
mean_test_price = y2.mean()
mae_test = mean_absolute_error(y2, y2_pred)
print(f"Final Test MAE (% of mean price): {mae_test / mean_test_price * 100:.2f}%")

"""# svr model"""

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# Train SVR model
svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)

# Predict
y_train_pred = svr_model.predict(X_train_scaled)
y_validation_pred = svr_model.predict(X_validation_scaled)
y2_pred = svr_model.predict(X2_scaled)

# Evaluation on training and validation sets
print("🔵 SVR (kernel='rbf', C=100, epsilon=0.1)")
print("Training R² Score:", r2_score(y_train, y_train_pred))
print("Validation R² Score:", r2_score(y_validation, y_validation_pred))
print("Validation MSE:", mean_squared_error(y_validation, y_validation_pred))
print("Validation MAE:", mean_absolute_error(y_validation, y_validation_pred))

# Relative MAE on validation
mean_val_price = y_validation.mean()
mae_val = mean_absolute_error(y_validation, y_validation_pred)
print(f"Validation MAE (% of mean price): {mae_val / mean_val_price * 100:.2f}%")

# Evaluation on final test set
print("\n--- Final Test Set (test_cleaned) ---")
print("Final Test R² Score:", r2_score(y2, y2_pred))
print("Final Test MSE:", mean_squared_error(y2, y2_pred))
print("Final Test MAE:", mean_absolute_error(y2, y2_pred))

# Relative MAE on test
mean_test_price = y2.mean()
mae_test = mean_absolute_error(y2, y2_pred)
print(f"Final Test MAE (% of mean price): {mae_test / mean_test_price * 100:.2f}%")

# Train SVR model
svr_model = SVR(kernel='rbf', C=10, epsilon=0.2)
svr_model.fit(X_train_scaled, y_train)

# Predict
y_train_pred = svr_model.predict(X_train_scaled)
y_validation_pred = svr_model.predict(X_validation_scaled)
y2_pred = svr_model.predict(X2_scaled)

# Evaluation on training and validation sets
print("🔵 SVR (kernel='rbf', C=10, epsilon=0.2)")
print("Training R² Score:", r2_score(y_train, y_train_pred))
print("Validation R² Score:", r2_score(y_validation, y_validation_pred))
print("Validation MSE:", mean_squared_error(y_validation, y_validation_pred))
print("Validation MAE:", mean_absolute_error(y_validation, y_validation_pred))

# Relative MAE on validation
mean_val_price = y_validation.mean()
mae_val = mean_absolute_error(y_validation, y_validation_pred)
print(f"Validation MAE (% of mean price): {mae_val / mean_val_price * 100:.2f}%")

# Evaluation on final test set
print("\n--- Final Test Set (test_cleaned) ---")
print("Final Test R² Score:", r2_score(y2, y2_pred))
print("Final Test MSE:", mean_squared_error(y2, y2_pred))
print("Final Test MAE:", mean_absolute_error(y2, y2_pred))

# Relative MAE on test
mean_test_price = y2.mean()
mae_test = mean_absolute_error(y2, y2_pred)
print(f"Final Test MAE (% of mean price): {mae_test / mean_test_price * 100:.2f}%")

# Train SVR model
svr_model = SVR(kernel='rbf', C=300, epsilon=0.05)
svr_model.fit(X_train_scaled, y_train)

# Predict
y_train_pred = svr_model.predict(X_train_scaled)
y_validation_pred = svr_model.predict(X_validation_scaled)
y2_pred = svr_model.predict(X2_scaled)

# Evaluation on training and validation sets
print("🔵 SVR (kernel='rbf', C=300, epsilon=0.05)")
print("Training R² Score:", r2_score(y_train, y_train_pred))
print("Validation R² Score:", r2_score(y_validation, y_validation_pred))
print("Validation MSE:", mean_squared_error(y_validation, y_validation_pred))
print("Validation MAE:", mean_absolute_error(y_validation, y_validation_pred))

# Relative MAE on validation
mean_val_price = y_validation.mean()
mae_val = mean_absolute_error(y_validation, y_validation_pred)
print(f"Validation MAE (% of mean price): {mae_val / mean_val_price * 100:.2f}%")

# Evaluation on final test set
print("\n--- Final Test Set (test_cleaned) ---")
print("Final Test R² Score:", r2_score(y2, y2_pred))
print("Final Test MSE:", mean_squared_error(y2, y2_pred))
print("Final Test MAE:", mean_absolute_error(y2, y2_pred))

# Relative MAE on test
mean_test_price = y2.mean()
mae_test = mean_absolute_error(y2, y2_pred)
print(f"Final Test MAE (% of mean price): {mae_test / mean_test_price * 100:.2f}%")

"""# random forest"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint


# Randomized search parameter grid
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 15),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Initialize and tune Random Forest
rf_model = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train_scaled, y_train)
best_rf_model = random_search.best_estimator_

# Fit and predict
best_rf_model.fit(X_train_scaled, y_train)
y_train_pred = best_rf_model.predict(X_train_scaled)
y_val_pred = best_rf_model.predict(X_validation_scaled)
y2_pred = best_rf_model.predict(X2_scaled)

# Evaluation
print("🟡 Random Forest Regression (with RandomizedSearchCV)")
print("Best Parameters:", random_search.best_params_)

# Train
print("\nTraining R² Score:", r2_score(y_train, y_train_pred))
print("Training MSE:", mean_squared_error(y_train, y_train_pred))
print("Training MAE:", mean_absolute_error(y_train, y_train_pred))
print(f"Training MAE (% of mean): {mean_absolute_error(y_train, y_train_pred) / y_train.mean() * 100:.2f}%")

# Validation
print("\nValidation R² Score:", r2_score(y_validation, y_val_pred))
print("Validation MSE:", mean_squared_error(y_validation, y_val_pred))
print("Validation MAE:", mean_absolute_error(y_validation, y_val_pred))
print(f"Validation MAE (% of mean): {mean_absolute_error(y_validation, y_val_pred) / y_validation.mean() * 100:.2f}%")

# Final test
print("\n--- Final Test Set (test_cleaned) ---")
print("Final Test R² Score:", r2_score(y2, y2_pred))
print("Final Test MSE:", mean_squared_error(y2, y2_pred))
print("Final Test MAE:", mean_absolute_error(y2, y2_pred))
print(f"Final Test MAE (% of mean): {mean_absolute_error(y2, y2_pred) / y2.mean() * 100:.2f}%")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint

# Prepare features and target
selected_features = ['Airbags', 'Prod. year', 'Mileage', 'Drive wheels', 'Wheel',
                     'Gear box type', 'Manufacturer', 'Model', 'Levy', 'Fuel type',
                     'Leather interior', 'Category', 'Cylinders']

X = train_cleaned[selected_features]
y = train_cleaned['Price'].astype(int)
X2 = test_cleaned[selected_features]
y2 = test_cleaned['Price'].astype(int)

# Split train_cleaned into training and validation sets
X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X2_scaled = scaler.transform(X2)  # Final test set

# Updated hyperparameter grid
param_dist = {
    'n_estimators': randint(150, 500),
    'max_depth': randint(10, 30),
    'min_samples_split': randint(2, 15),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True]
}

# Initialize and tune Random Forest
rf_model = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train_scaled, y_train)
best_rf_model = random_search.best_estimator_

# Fit and predict
best_rf_model.fit(X_train_scaled, y_train)
y_train_pred = best_rf_model.predict(X_train_scaled)
y_val_pred = best_rf_model.predict(X_validation_scaled)
y2_pred = best_rf_model.predict(X2_scaled)

# Evaluation
print("🟡 Random Forest Regression (with Updated Hyperparameters)")
print("Best Parameters:", random_search.best_params_)

# Train
print("\nTraining R² Score:", r2_score(y_train, y_train_pred))
print("Training MSE:", mean_squared_error(y_train, y_train_pred))
print("Training MAE:", mean_absolute_error(y_train, y_train_pred))
print(f"Training MAE (% of mean): {mean_absolute_error(y_train, y_train_pred) / y_train.mean() * 100:.2f}%")

# Validation
print("\nValidation R² Score:", r2_score(y_validation, y_val_pred))
print("Validation MSE:", mean_squared_error(y_validation, y_val_pred))
print("Validation MAE:", mean_absolute_error(y_validation, y_val_pred))
print(f"Validation MAE (% of mean): {mean_absolute_error(y_validation, y_val_pred) / y_validation.mean() * 100:.2f}%")

# Final test
print("\n--- Final Test Set (test_cleaned) ---")
print("Final Test R² Score:", r2_score(y2, y2_pred))
print("Final Test MSE:", mean_squared_error(y2, y2_pred))
print("Final Test MAE:", mean_absolute_error(y2, y2_pred))
print(f"Final Test MAE (% of mean): {mean_absolute_error(y2, y2_pred) / y2.mean() * 100:.2f}%")

"""# xgb"""

# Train Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Predict
y_train_pred = gb_model.predict(X_train_scaled)
y_validation_pred = gb_model.predict(X_validation_scaled)
y2_pred = gb_model.predict(X2_scaled)  # Final prediction on test_cleaned

# Evaluation Metrics
print("Training R² Score:", r2_score(y_train, y_train_pred))
print("Validation R² Score:", r2_score(y_validation, y_validation_pred))
print("Validation MSE:", mean_squared_error(y_validation, y_validation_pred))
print("Validation MAE:", mean_absolute_error(y_validation, y_validation_pred))

# Relative MAE for validation set
mean_val_price = y_validation.mean()
mae_val = mean_absolute_error(y_validation, y_validation_pred)
print(f"Validation MAE (% of mean price): {mae_val / mean_val_price * 100:.2f}%")

print("\n--- Final Test Set (test_cleaned) ---")
print("Final Test R² Score:", r2_score(y2, y2_pred))
print("Final Test MSE:", mean_squared_error(y2, y2_pred))
print("Final Test MAE:", mean_absolute_error(y2, y2_pred))

# Relative MAE for final test set
mean_test_price = y2.mean()
mae_test = mean_absolute_error(y2, y2_pred)
print(f"Final Test MAE (% of mean price): {mae_test / mean_test_price * 100:.2f}%")

# Train SVR model
svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)

# Predict
y_train_pred = svr_model.predict(X_train_scaled)
y_validation_pred = svr_model.predict(X_validation_scaled)
y2_pred = svr_model.predict(X2_scaled)  # Final prediction on test_cleaned

# Evaluation Metrics
print("Training R² Score:", r2_score(y_train, y_train_pred))
print("Validation R² Score:", r2_score(y_validation, y_validation_pred))
print("Validation MSE:", mean_squared_error(y_validation, y_validation_pred))
print("Validation MAE:", mean_absolute_error(y_validation, y_validation_pred))

# Relative MAE for validation set
mean_val_price = y_validation.mean()
mae_val = mean_absolute_error(y_validation, y_validation_pred)
print(f"Validation MAE (% of mean price): {mae_val / mean_val_price * 100:.2f}%")

print("\n--- Final Test Set (test_cleaned) ---")
print("Final Test R² Score:", r2_score(y2, y2_pred))
print("Final Test MSE:", mean_squared_error(y2, y2_pred))
print("Final Test MAE:", mean_absolute_error(y2, y2_pred))

# Relative MAE for final test set
mean_test_price = y2.mean()
mae_test = mean_absolute_error(y2, y2_pred)
print(f"Final Test MAE (% of mean price): {mae_test / mean_test_price * 100:.2f}%")

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Train Neural Network
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50),   # you can tune this
                         activation='relu',              # try 'tanh' or 'logistic' as alternatives
                         solver='adam',
                         max_iter=1000,
                         random_state=42)

mlp_model.fit(X_train_scaled, y_train)

# Predict
y_train_pred = mlp_model.predict(X_train_scaled)
y_validation_pred = mlp_model.predict(X_validation_scaled)
y2_pred = mlp_model.predict(X2_scaled)  # Final prediction on test_cleaned

# Evaluation Metrics
print("Neural Network Training R² Score:", r2_score(y_train, y_train_pred))
print("Neural Network Validation R² Score:", r2_score(y_validation, y_validation_pred))
print("Neural Network Validation MSE:", mean_squared_error(y_validation, y_validation_pred))
print("Neural Network Validation MAE:", mean_absolute_error(y_validation, y_validation_pred))

# Relative MAE for validation set
mean_val_price = y_validation.mean()
mae_val = mean_absolute_error(y_validation, y_validation_pred)
print(f"Validation MAE (% of mean price): {mae_val / mean_val_price * 100:.2f}%")

print("\n--- Final Test Set (test_cleaned) ---")
print("Final Test R² Score:", r2_score(y2, y2_pred))
print("Final Test MSE:", mean_squared_error(y2, y2_pred))
print("Final Test MAE:", mean_absolute_error(y2, y2_pred))

# Relative MAE for final test set
mean_test_price = y2.mean()
mae_test = mean_absolute_error(y2, y2_pred)
print(f"Final Test MAE (% of mean price): {mae_test / mean_test_price * 100:.2f}%")

