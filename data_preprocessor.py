import pandas as pd
import os

# Configuration
PATH = 'Missioni_AGV_B.csv'
OUTPUT_FOLDER = 'Data'
OUTPUT_FILE = 'data_cleaned.csv'
FEATURES_MODEL = ['DataOraInizio', 'DataOraFine', 'DurataMissione', 
                  'NodoOrigine', 'NodoDestinazione', 'DataOraInserimento', 
                  'Operazione', 'Agv']
DATE_COLUMNS = ['DataOraInizio', 'DataOraFine', 'DataOraInserimento']
MISSING_VALUE_THRESHOLD = 0.7  # Keep columns with at least 70% of data


def load_and_clean_data(path):
    """Load and clean the dataset."""
    # Load dataset with optimized dtypes for date columns
    data = pd.read_csv(path)
    
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Remove columns with more than 30% missing values
    data = data.dropna(thresh=len(data) * MISSING_VALUE_THRESHOLD, axis=1)
    
    # Drop rows with remaining missing values
    data = data.dropna()
    
    # Convert date columns to datetime format efficiently
    for col in DATE_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], format='mixed')
    
    return data


def create_features(data):
    """Create engineered features."""
    # Select relevant features and copy
    data_cleaned = data[FEATURES_MODEL].copy()
    
    # Calculate delay between insertion time and mission start time (in seconds)
    data_cleaned['Delay_Insertion_to_Start'] = (
        data_cleaned['DataOraInizio'] - data_cleaned['DataOraInserimento']
    ).dt.total_seconds()
    
    return data_cleaned


def save_data(data, output_folder=OUTPUT_FOLDER, output_file=OUTPUT_FILE):
    """Save processed data to folder."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_folder, output_file)
    data.to_csv(output_path, index=False)
    print(f"Data saved successfully to: {output_path}")
    
    return output_path


if __name__ == '__main__':
    # Load and preprocess data
    data = load_and_clean_data(PATH)
    data_cleaned = create_features(data)
    
    # Save processed data
    save_data(data_cleaned)