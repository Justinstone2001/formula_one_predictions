import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures
from PIL import Image, UnidentifiedImageError

# Load the combined data to get the mappings
combined_data = pd.read_csv("combined_data.csv")

# Load the saved model
best_model = joblib.load('f1_finishing_position_predictor.pkl')

# Load the saved polynomial features transformer
poly = joblib.load('poly_transformer.pkl')

# Load the features columns
features_columns = joblib.load('features_columns.pkl')

# Function to safely get values
def safe_get_value(series, default_value=0):
    try:
        return series.values[0]
    except IndexError:
        return default_value

# Function to predict finishing position
def predict_finishing_position(track, driver, starting_grid):
    input_data = {
        'Track': [track],
        'Driver': [driver],
        'Starting Grid': [starting_grid]
    }

    input_df = pd.DataFrame(input_data)
    input_df['driver_avg_starting_position'] = safe_get_value(combined_data.groupby('Driver')['Starting Grid'].transform('mean').loc[combined_data['Driver'] == driver])
    input_df['driver_avg_finishing_position'] = safe_get_value(combined_data.groupby('Driver')['Position'].transform('mean').loc[combined_data['Driver'] == driver])
    input_df['team_avg_starting_position'] = safe_get_value(combined_data.groupby('Team')['Starting Grid'].transform('mean').loc[combined_data['Driver'] == driver])
    input_df['team_avg_finishing_position'] = safe_get_value(combined_data.groupby('Team')['Position'].transform('mean').loc[combined_data['Driver'] == driver])
    input_df['track_driver_avg_finishing_position'] = safe_get_value(combined_data.groupby(['Track', 'Driver'])['Position'].transform('mean').loc[(combined_data['Track'] == track) & (combined_data['Driver'] == driver)])
    input_df['track_team_avg_finishing_position'] = safe_get_value(combined_data.groupby(['Track', 'Team'])['Position'].transform('mean').loc[(combined_data['Track'] == track) & (combined_data['Driver'] == driver)])
    input_df['driver_form'] = safe_get_value(combined_data.groupby('Driver', group_keys=False)['Position'].apply(lambda x: x.rolling(10, min_periods=1).mean()).loc[combined_data['Driver'] == driver])
    input_df['team_form'] = safe_get_value(combined_data.groupby('Team', group_keys=False)['Position'].apply(lambda x: x.rolling(10, min_periods=1).mean()).loc[combined_data['Driver'] == driver])
    input_df['driver_win_count'] = safe_get_value(combined_data.groupby('Driver')['Position'].transform(lambda x: (x == 1).sum()).loc[combined_data['Driver'] == driver])
    input_df['team_win_count'] = safe_get_value(combined_data.groupby('Team')['Position'].transform(lambda x: (x == 1).sum()).loc[combined_data['Driver'] == driver])
    input_df['driver_podium_count'] = safe_get_value(combined_data.groupby('Driver')['Position'].transform(lambda x: (x <= 3).sum()).loc[combined_data['Driver'] == driver])
    input_df['team_podium_count'] = safe_get_value(combined_data.groupby('Team')['Position'].transform(lambda x: (x <= 3).sum()).loc[combined_data['Driver'] == driver])
    input_df['driver_team_interaction'] = safe_get_value(combined_data.groupby(['Driver', 'Team'])['Position'].transform('mean').loc[(combined_data['Driver'] == driver) & (combined_data['Track'] == track)])
    input_df['track_starting_grid_interaction'] = safe_get_value(combined_data.groupby(['Track', 'Starting Grid'])['Position'].transform('mean').loc[(combined_data['Track'] == track) & (combined_data['Starting Grid'] == starting_grid)])
    input_df['driver_track_interaction'] = safe_get_value(combined_data.groupby(['Driver', 'Track'])['Position'].transform('mean').loc[(combined_data['Driver'] == driver) & (combined_data['Track'] == track)])
    input_df['team_track_interaction'] = safe_get_value(combined_data.groupby(['Team', 'Track'])['Position'].transform('mean').loc[(combined_data['Track'] == track) & (combined_data['Driver'] == driver)])

    input_df = pd.get_dummies(input_df, drop_first=True)

    missing_cols = set(features_columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[features_columns]

    input_features_poly = poly.transform(input_df)

    prediction = best_model.predict(input_features_poly)
    
    rounded_prediction = round(prediction[0])
    
    return rounded_prediction

# Streamlit app

# Display title with logo
logo, title = st.columns([1.5, 4])
with logo:
    st.image("logo/f1_logo.png", width=100)
with title:
    st.markdown("""
        <div style="font-size: 35px; color: red; font-family: 'Formula 1';">
            F1 Predictive Model
        </div>
    """, unsafe_allow_html=True)

# List of drivers
drivers_list = {
    'Alpine': ['Pierre Gasly', 'Esteban Ocon'],
    'Aston Martin': ['Fernando Alonso', 'Lance Stroll'],
    'Ferrari': ['Charles Leclerc', 'Carlos Sainz'],
    'Haas': ['Nico Hulkenberg', 'Kevin Magnussen'],
    'McLaren': ['Lando Norris', 'Oscar Piastri'],
    'Mercedes': ['Lewis Hamilton', 'George Russell'],
    'Red Bull': ['Sergio Perez', 'Max Verstappen'],
    'Sauber (Alfa Romeo)': ['Valtteri Bottas', 'Zhou Guanyu'],
    'AlphaTauri': ['Daniel Ricciardo', 'Yuki Tsunoda'],
    'Williams': ['Alex Albon', 'Logan Sargeant']
}

# Flatten the list of drivers
drivers = [driver for team in drivers_list.values() for driver in team]

tracks = combined_data['Track'].unique()

track = st.selectbox('Select Track', tracks)

track_image_path = f'track_images/{track.lower().replace(" ", "_")}.jpeg'  # Ensure the track names match file names
try:
    track_image = Image.open(track_image_path)
    st.image(track_image, caption=f"{track} Race Track", use_column_width=True)
except FileNotFoundError:
    st.write(f"Image for {track} not found.")


driver = st.selectbox('Select Driver', drivers)

driver_last_name = driver.split()[-1].lower()
driver_image_path = f'driver_images/{driver_last_name}.jpeg'  # Ensure the driver last names match file names

try:
    driver_image = Image.open(driver_image_path)
    st.image(driver_image, caption=f"{driver}", use_column_width=True, width=200)
except FileNotFoundError:
    st.write(f"Image for {driver} not found.")
except UnidentifiedImageError:
    st.write(f"Cannot identify image file: {driver_image_path}")

starting_grid = st.slider('Select Starting Grid Position', 1, 20, 1)


if st.button('Predict'):
    predicted_position = predict_finishing_position(track, driver, starting_grid)
    st.markdown(f"""
        <div style="text-align: center; font-size: 24px;">
            <strong>{driver}</strong>
        </div>
        <div style="text-align: center; font-size: 60px; color: red; font-family: 'Formula 1';">
            {predicted_position}
        </div>
    """, unsafe_allow_html=True)
