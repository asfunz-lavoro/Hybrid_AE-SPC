import pandas as pd
import os

# Configuration
PATH = 'Missioni_AGV_B.csv'
OUTPUT_FOLDER = 'Data'
OUTPUT_FILE = 'data_cleaned.csv'
FEATURES_MODEL = ['DataOraInizio', 'DataOraFine', 'DurataMissione', 
                  'NodoOrigine', 'NodoDestinazione', 'DataOraInserimento', 
                  'Operazione', 'Agv', 'TerminataConSuccesso']
DATE_COLUMNS = ['DataOraInizio', 'DataOraFine', 'DataOraInserimento']
MISSING_VALUE_THRESHOLD = 0.7  # Keep columns with at least 70% of data


def load(path):
    """Loadthe dataset."""
    # Load dataset with optimized dtypes for date columns
    data = pd.read_csv(path)
    
    # Convert date columns to datetime format efficiently
    for col in DATE_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], format='mixed')
    
    return data

def data_cleaning(data, threshold=MISSING_VALUE_THRESHOLD):
    """Clean the dataset by handling missing values."""
    # Calculate the threshold for missing values
    min_non_missing = int((1 - threshold) * len(data))
    
    # Drop columns with too many missing values
    data_cleaned = data.dropna(axis=1, thresh=min_non_missing)
    
    # Drop rows with any remaining missing values
    data_cleaned = data_cleaned.dropna(axis=0)

    return data_cleaned


def create_features(data):
    """Create engineered features."""
    # Select relevant features and copy
    data = data[FEATURES_MODEL].copy()
    
    # Calculate delay between insertion time and mission start time (in seconds)
    data['Delay_Insertion_to_Start'] = (
        data['DataOraInizio'] - data['DataOraInserimento']
    ).dt.total_seconds()

    return data

def save_data(data, output_folder=OUTPUT_FOLDER, output_file=OUTPUT_FILE):
    """Save processed data to folder."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_folder, output_file)
    data.to_csv(output_path, index=False)
    print(f"Data saved successfully to: {output_path}")
    
    return output_path


def save_data_by_agv(data, output_folder=OUTPUT_FOLDER):
    """Save processed data to separate CSV files for each AGV."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get unique AGVs
    agvs = data['Agv'].unique()
    saved_files = []
    
    # Save data for each AGV
    for agv in agvs:
        agv_data = data[data['Agv'] == agv]
        
        # Create filename for AGV
        output_file = f'data_cleaned_{agv}.csv'
        output_path = os.path.join(output_folder, output_file)
        
        # Save to CSV
        agv_data.to_csv(output_path, index=False)
        saved_files.append(output_path)
        print(f"Data for AGV {agv} saved successfully to: {output_path} ({len(agv_data)} rows)")
    
    return saved_files


if __name__ == '__main__':
    # Load and preprocess data
    data = load(PATH)
    data = data_cleaning(data)
    data = create_features(data)
    
    # Save options:
    # Option 1: Save all data in a single file
    #save_data(data)
    
    # Option 2: Save separate files for each AGV
    save_data_by_agv(data)