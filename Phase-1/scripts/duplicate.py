import pandas as pd

def remove_duplicates_and_sort(input_file, output_file):
    # Load the GNSS data
    df = pd.read_csv(input_file)
    
    # Convert Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Remove duplicate rows based on Latitude and Longitude
    df = df.drop_duplicates(subset=['Latitude', 'Longitude'])
    
    # Sort the DataFrame by Timestamp
    df_sorted = df.sort_values(by='Timestamp')
    
    # Save the cleaned and sorted DataFrame to a new CSV file
    df_sorted.to_csv(output_file, index=False)
    print(f"Cleaned and sorted CSV saved to {output_file}")

if __name__ == "__main__":
    # File paths
    input_file = '../Location_data/Rahil/Data2/Rahil_gnss_data2.csv'  # Replace with your actual file path
    output_file = '../Location_data/Rahil/Data2/Rahil_sorted_dub_gnss_data2.csv'  # Replace with your desired output file path
    
    # Remove duplicates and sort the CSV file by Timestamp
    remove_duplicates_and_sort(input_file, output_file)