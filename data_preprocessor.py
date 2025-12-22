import pandas as pd
import os

# --- Configuration (Kept for easy modification outside the class) ---
PATH = 'Missioni_AGV_B.csv'
OUTPUT_FOLDER = 'Data'
OUTPUT_FILE = 'data_cleaned.csv'
FEATURES_MODEL = ['DataOraInizio', 'DataOraFine', 'DurataMissione', 
                  'NodoOrigine', 'NodoDestinazione', 'DataOraInserimento', 
                  'Operazione', 'Agv', 'TerminataConSuccesso']
DATE_COLUMNS = ['DataOraInizio', 'DataOraFine', 'DataOraInserimento']
MISSING_VALUE_THRESHOLD = 0.7  # Keep columns with at least 70% of data


class AGVDataPreprocessor:
    """
    A class for loading, cleaning, feature engineering, and saving AGV mission data.
    """

    def __init__(self, path=PATH, output_folder=OUTPUT_FOLDER, 
                 output_file=OUTPUT_FILE, features_model=FEATURES_MODEL, 
                 date_columns=DATE_COLUMNS, missing_value_threshold=MISSING_VALUE_THRESHOLD):
        """
        Initializes the Preprocessor with configuration parameters.
        """
        self.path = path
        self.output_folder = output_folder
        self.output_file = output_file
        self.features_model = features_model
        self.date_columns = date_columns
        self.missing_value_threshold = missing_value_threshold
        self.data = None # Store the processed DataFrame


    def load_data(self):
        """
        Load the dataset from the specified path and convert date columns.
        The initial feature selection is done here to work with less data sooner.
        """
        print(f"Loading data from: {self.path}")
        # Load dataset
        try:
            data = pd.read_csv(self.path)
        except FileNotFoundError:
            print(f"Error: File not found at {self.path}")
            return None
        
        # Select necessary columns for processing/modeling
        data = data[self.features_model].copy()
        
        # Convert date columns to datetime format efficiently
        for col in self.date_columns:
            if col in data.columns:
                # 'mixed' format infers the best format for each column
                data[col] = pd.to_datetime(data[col], format='mixed', errors='coerce')
        
        self.data = data
        return self.data


    def clean_data(self):
        """
        Clean the dataset by handling missing values based on the threshold.
        """
        if self.data is None:
            print("Error: Data not loaded. Call load_data() first.")
            return None
            
        print(f"Cleaning data (initial shape: {self.data.shape})")
        data = self.data.copy()

        # Calculate the minimum number of non-missing values required to keep a column
        min_non_missing = int(self.missing_value_threshold * len(data))
        
        # Drop columns with too many missing values (too many NaN values)
        data_cleaned = data.dropna(axis=1, thresh=min_non_missing)
        
        # Drop rows with any remaining missing values (row-wise drop)
        # This handles cases where a few key columns might still have NAs after the column drop
        initial_rows = len(data_cleaned)
        data_cleaned = data_cleaned.dropna(axis=0)
        dropped_rows = initial_rows - len(data_cleaned)

        print(f"Dropped {dropped_rows} rows with remaining missing values.")
        print(f"Data cleaned shape: {data_cleaned.shape}")

        self.data = data_cleaned
        return self.data


    def create_features(self):
        """
        Create engineered features, specifically the 'Delay_Insertion_to_Start'.
        """
        if self.data is None:
            print("Error: Data not loaded/cleaned. Call load_data() and clean_data() first.")
            return None
            
        print("Creating engineered features...")
        data = self.data.copy()
        
        # Calculate delay between insertion time and mission start time (in seconds)
        # Using .dt.total_seconds() on the difference of two datetime series
        data['Delay_Insertion_to_Start'] = (
            data['DataOraInizio'] - data['DataOraInserimento']
        ).dt.total_seconds()

        self.data = data
        return self.data


    def save_data(self, output_file=None):
        """
        Save the processed data to a single CSV file.
        """
        if self.data is None:
            print("Error: Data not processed. Call the processing methods first.")
            return None
            
        final_output_file = output_file if output_file else self.output_file
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Save to CSV
        output_path = os.path.join(self.output_folder, final_output_file)
        self.data.to_csv(output_path, index=False)
        print(f"✅ Data saved successfully to: {output_path} ({len(self.data)} rows)")
        
        return output_path


    def save_data_by_agv(self):
        """
        Save processed data to separate CSV files for each unique AGV.
        """
        if self.data is None:
            print("Error: Data not processed. Call the processing methods first.")
            return None
            
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Get unique AGVs
        agvs = self.data['Agv'].unique()
        saved_files = []
        
        print(f"\nSaving data split by {len(agvs)} unique AGVs...")
        
        # Save data for each AGV
        for agv in agvs:
            agv_data = self.data[self.data['Agv'] == agv]
            
            # Create filename for AGV
            output_file = f'data_cleaned_{agv}.csv'
            output_path = os.path.join(self.output_folder, output_file)
            
            # Save to CSV
            agv_data.to_csv(output_path, index=False)
            saved_files.append(output_path)
            print(f"  > AGV {agv}: Saved to {output_path} ({len(agv_data)} rows)")
        
        print("✅ All AGV data saved.")
        return saved_files


    def run_pipeline(self, save_option='single'):
        """
        Execute the full preprocessing pipeline.
        'single' saves all data to one file.
        'split' saves data to separate files per AGV.
        """
        print("--- Starting AGV Data Preprocessing Pipeline ---")
        
        # 1. Load Data
        if self.load_data() is None:
            return
            
        # 2. Clean Data
        if self.clean_data() is None:
            return
            
        # 3. Create Features (Feature Engineering)
        if self.create_features() is None:
            return
            
        # 4. Save Data
        if save_option == 'single':
            self.save_data()
        elif save_option == 'split':
            self.save_data_by_agv()
        else:
            print(f"Warning: Unknown save option '{save_option}'. Data was processed but not saved.")

        print("--- Pipeline Complete ---")


if __name__ == '__main__':
    # Initialize the preprocessor with default/custom configuration
    preprocessor = AGVDataPreprocessor()
    
    # Run the entire pipeline
    # Choose save_option='single' to save all data to one file (data_cleaned.csv)
    # Choose save_option='split' to save data to separate files for each AGV (data_cleaned_AGV1.csv, etc.)
    preprocessor.run_pipeline(save_option='split')