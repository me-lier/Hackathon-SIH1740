import pandas as pd

def sort_csv_by_timestamp(input_file, output_file):
    # Load the GNSS data
    df = pd.read_csv(input_file)
    
    # Convert Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Sort the DataFrame by Timestamp
    df_sorted = df.sort_values(by='Timestamp')
    
    # Save the sorted DataFrame to a new CSV file
    df_sorted.to_csv(output_file, index=False)
    print(f"Sorted CSV saved to {output_file}")

if __name__ == "__main__":
    # File paths
    input_file = '../Location_data/Rahil/Data2/Rahil_gnss_data2.csv'
    output_file = '../Location_data/Rahil/Data2/Rahil_sorted_gnss_data2.csv'
    
    # Sort the CSV file by Timestamp
    sort_csv_by_timestamp(input_file, output_file)
