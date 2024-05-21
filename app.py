import pickle

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

# Directory to save CSV files
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)
consolidated_consumption_file = os.path.join(data_dir, 'consolidated_consumption_data.csv')
consolidated_generation_file = os.path.join(data_dir, 'consolidated_generation_data.csv')


# Function to consolidate data from multiple CSV files
def consolidate_data(file_path, data_dir, file_prefix):
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith(file_prefix) and f.endswith('.csv')]
    df_list = [pd.read_csv(file) for file in all_files]
    if df_list:
        consolidated_df = pd.concat(df_list, ignore_index=True)
        consolidated_df.to_csv(file_path, index=False)
    else:
        columns = ['city', 'month', 'energy_consumed'] if 'consumption' in file_prefix else ['plant', 'month', 'energy_generated']
        consolidated_df = pd.DataFrame(columns=columns)
        consolidated_df.to_csv(file_path, index=False)

# Consolidate initial data
consolidate_data(consolidated_consumption_file, data_dir, 'consumption')
consolidate_data(consolidated_generation_file, data_dir, 'generation')

def load_generation_capacities():
    # Load generation capacities from CSV file
    capacities_df = pd.read_csv('data/generation_capacities.csv')
    generation_capacities = capacities_df.set_index('plant')['capacity'].to_dict()
    return generation_capacities

# Load data for training
# Load data for training
consumption_df = pd.read_csv(consolidated_consumption_file)
consumption_df['month'] = pd.to_datetime(consumption_df['month'], format='%B').dt.month

X = consumption_df[['city', 'month']]
X = pd.get_dummies(X, columns=['city'])
y = consumption_df['energy_consumed']

# Check if data is available for splitting
if X.empty or y.empty:
    print("Error: Insufficient data for training the model.")
else:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model
    with open('energy_model.pkl', 'wb') as f:
        pickle.dump(model, f)




@app.route('/collect_consumption_data', methods=['POST'])
def collect_consumption_data():
    data_list = request.json  # List of consumption data
    if not isinstance(data_list, list):
        return jsonify({'error': 'Input data should be a list of dictionaries'}), 400

    for data in data_list:
        city = data['city']
        month = data['month']
        energy_consumed = data['energy_consumed']

        # Create a DataFrame
        df = pd.DataFrame([{
            'city': city,
            'month': month,
            'energy_consumed': energy_consumed
        }])

        # Save data to CSV
        file_path = os.path.join(data_dir, f'consumption_{city}_{month}.csv')
        df.to_csv(file_path, index=False, mode='a', header=not os.path.exists(file_path))

    # Consolidate data
    consolidate_data(consolidated_consumption_file, data_dir, 'consumption')

    return jsonify({'message': 'Consumption data collected successfully'})

@app.route('/collect_generation_data', methods=['POST'])
def collect_generation_data():
    data_list = request.json  # List of generation data
    if not isinstance(data_list, list):
        return jsonify({'error': 'Input data should be a list of dictionaries'}), 400

    for data in data_list:
        plant = data['plant']
        month = data['month']
        energy_generated = data['energy_generated']

        # Create a DataFrame
        df = pd.DataFrame([{
            'plant': plant,
            'month': month,
            'energy_generated': energy_generated
        }])

        # Save data to CSV
        file_path = os.path.join(data_dir, f'generation_{plant}_{month}.csv')
        df.to_csv(file_path, index=False, mode='a', header=not os.path.exists(file_path))

    # Consolidate data
    consolidate_data(consolidated_generation_file, data_dir, 'generation')

    return jsonify({'message': 'Generation data collected successfully'})

@app.route('/distribute_energy', methods=['POST'])
def distribute_energy():
    data = request.json
    cities = data.get('cities')
    month = data.get('month')

    # Input validation
    if not cities or not month:
        return jsonify({'error': 'Cities and month are required parameters'}), 400

    try:
        month_num = pd.to_datetime(month, format='%B').month
    except ValueError:
        return jsonify({'error': 'Invalid month format'}), 400

    distribution_plans = []
    for city in cities:
        # Predict energy consumption
        input_data = pd.DataFrame({'city': [city], 'month': [month_num]})
        input_data = pd.get_dummies(input_data, columns=['city'])
        missing_features = set(model.feature_names_in_) - set(input_data.columns)
        for feature in missing_features:
            input_data[feature] = 0
        input_data = input_data[model.feature_names_in_]

        predicted_consumption = model.predict(input_data)[0]

        # Load generation data
        generation_df = pd.read_csv(consolidated_generation_file)
        total_generated_energy = generation_df.loc[
            (generation_df['plant'] == city) & (generation_df['month'] == month_num), 'energy_generated'
        ].sum()

        # Create distribution plan
        distribution_plan = {
            'city': city,
            'month': month,
            'predicted_consumption': predicted_consumption,
            'total_generated_energy': total_generated_energy
        }

        if total_generated_energy >= predicted_consumption:
            distribution_plan['status'] = 'Sufficient energy available'
        else:
            distribution_plan['status'] = 'Insufficient energy, load shedding required'
            distribution_plan['shortfall'] = predicted_consumption - total_generated_energy

        # Allocate energy from plants
        allocated_energy = {}
        generation_capacities = load_generation_capacities()
        for plant, capacity in generation_capacities.items():
            if total_generated_energy >= predicted_consumption:
                allocated_energy[plant] = 0
            else:
                allocated = min(capacity, distribution_plan['shortfall'])
                allocated_energy[plant] = int(allocated)
                distribution_plan['shortfall'] -= allocated

        distribution_plan['allocated_energy'] = allocated_energy

        # Calculate accuracy metrics
        mae = mean_absolute_error([total_generated_energy], [predicted_consumption])
        rmse = np.sqrt(mean_squared_error([total_generated_energy], [predicted_consumption]))

        mae_percentage = (mae / total_generated_energy) * 100 if total_generated_energy != 0 else 0
        rmse_percentage = (rmse / total_generated_energy) * 100 if total_generated_energy != 0 else 0

        accuracy_metrics = {
            'Mean Absolute Error (MAE) Percentage': mae_percentage,
            'Root Mean Squared Error (RMSE) Percentage': rmse_percentage
        }

        distribution_plan['accuracy_metrics'] = accuracy_metrics
        distribution_plans.append(distribution_plan)

    return jsonify(distribution_plans)

if __name__ == '__main__':
    app.run(debug=True)
