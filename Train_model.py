import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Show full width of columns and full content in each cell
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", None)        # Don't break line based on width
pd.set_option("display.max_colwidth", None) # Show full column content

# Load the dataset
data = pd.read_csv("House_prices.csv")  # Update filename if different

# Print shape and head
# print("Shape:", data.shape)
# print("Columns:", data.columns.tolist())

# Convert Excel serial date to datetime
data["Date"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(data["Date"], unit="D")

# Check data types of all columns
# print("DataTypes of All coulmns")
# print(data.dtypes)

# Check for missing values
# print("\nMissing Values in Each Column:")
# print(data.isnull().sum().sort_values(ascending=False))



#
#

# Drop columns not useful for prediction
data = data.drop(["id", "Date"], axis=1)  # Already used or irrelevant

# Define features and target
X = data.drop("Price", axis=1)
y = data["Price"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





# # .....................Linear LinearRegression Model training and evalutaion...............


# from sklearn.linear_model import LinearRegression

# # Train a Linear Regression model
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Predict
# predictions = model.predict(X_test)
#
#
# # Evaluate predictions
# mse = mean_squared_error(y_test, predictions)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, predictions)
#
#
# print("\n\n")
# print("Model: Linear Regression")
# print(f"RMSE: {rmse}")
# print(f"R² Score: {r2}")
#
#
#
# # ......................Descion Tree Model Training and Evaluation.............
# from sklearn.tree import DecisionTreeRegressor
#
# # Train the model
# dt_model = DecisionTreeRegressor(random_state=42)
# dt_model.fit(X_train, y_train)
#
# # Predict
# dt_predictions = dt_model.predict(X_test)
#
# # Evaluate
# dt_rmse = np.sqrt(mean_squared_error(y_test, dt_predictions))
# dt_r2 = r2_score(y_test, dt_predictions)
#
#
# print("\n\n")
# print("Model: Decision Tree")
# print(f"RMSE: {dt_rmse}")
# print(f"R² Score: {dt_r2}")
#
#
#
#
# # ....................Random forest Regressor Model training and evaluation...........
#
#
#
#
# from sklearn.ensemble import RandomForestRegressor
#
# # Train the model
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)
#
# # Predict
# rf_predictions = rf_model.predict(X_test)
#
# # Evaluate
# rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
# rf_r2 = r2_score(y_test, rf_predictions)
#
#
# print("\n\n")
# print("Model: Random Forest")
# print(f"RMSE: {rf_rmse}")
# print(f"R² Score: {rf_r2}")
#
# ..................Gradient boosting regressor Model Training and Evaluation............


from sklearn.ensemble import GradientBoostingRegressor

# Train the model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Predict
gb_predictions = gb_model.predict(X_test)

# Evaluate
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_predictions))
gb_r2 = r2_score(y_test, gb_predictions)

# print("\n\n")
# print("Model: Gradient Boosting")
# print(f"RMSE: {gb_rmse}")
# print(f"R² Score: {gb_r2}")



# Save your trained model (Gradient Boosting)
joblib.dump(gb_model, "house_price_model.pkl")



# print(data.head(5))





