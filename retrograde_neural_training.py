
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset
btc_data = pd.read_csv('/mnt/data/BTC-USD.csv')

# Create lag and lead features
n_lags = 5
n_leads = 5
for i in range(1, n_lags + 1):
    btc_data[f'lag_{i}'] = btc_data['Close'].shift(i)
for i in range(1, n_leads + 1):
    btc_data[f'lead_{i}'] = btc_data['Close'].shift(-i)
btc_data = btc_data.dropna()

# Split the data
X_ant = btc_data[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']]
y_ant = btc_data['Close']
X_ret = btc_data[['lead_1', 'lead_2', 'lead_3', 'lead_4', 'lead_5']]
y_ret = btc_data['Close']
X_ant_train, X_ant_test, y_ant_train, y_ant_test = train_test_split(X_ant, y_ant, test_size=0.2, shuffle=False)
X_ret_train, X_ret_test, y_ret_train, y_ret_test = train_test_split(X_ret, y_ret, test_size=0.2, shuffle=False)

# Train the anterograde model
ant_model = LinearRegression()
ant_model.fit(X_ant_train, y_ant_train)
ant_predictions = ant_model.predict(X_ant_test)

# Train the adjusted retrograde model
ret_model_adjusted = LinearRegression()
ret_model_adjusted.fit(X_ret_train, y_ret_train)
ret_adjusted_predictions = ret_model_adjusted.predict(X_ret_test)

# Ensemble the predictions
ensemble_predictions = (ant_predictions + ret_adjusted_predictions) / 2

# Evaluation
ant_mae = mean_absolute_error(y_ant_test, ant_predictions)
ret_adjusted_mae = mean_absolute_error(y_ret_test, ret_adjusted_predictions)
ensemble_mae = mean_absolute_error(y_ant_test, ensemble_predictions)

print(f"Anterograde Model MAE: {ant_mae}")
print(f"Adjusted Retrograde Model MAE: {ret_adjusted_mae}")
print(f"Ensemble Model MAE: {ensemble_mae}")
