# Financial-Anomaly-DetectionFinancial-Anomaly-Detection
## Introduction
This project involves developing a robust system to identify and report anomalies in financial transaction datasets. The system reads transaction data, performs statistical analysis, detects anomalies based on established thresholds, and generates a comprehensive report of the detected anomalies. This project demonstrates the ability to handle large datasets, apply statistical methods for data analysis, and integrate various data processing techniques in Python.

## Requirements
- Python 3.7+
- Pandas
- NumPy

## Scripts
- preprocess.py: Script for data preprocessing.
- analysis.py: Script for performing statistical analysis.
- anomaly_detection.py: Script for detecting anomalies.
- report.py: Script for generating anomaly reports.
- main.py: Main orchestration script to run the entire process.

## Usage
Install the required packages:
- pip install -r requirements.txt
- Run the main script:
- python src/main.py

## Prerequisite
Ensure that the financial_transactions.csv dataset is placed in the data/ directory. Adjust the file paths in main.py accordingly.

## Approach
### Data Preprocessing
- Load and clean the dataset.
- Handle missing or corrupt entries.
### Statistical Analysis
- Calculate essential statistical metrics like mean and standard deviation for transaction amounts by type.
- Establish thresholds for detecting outliers using Z-score.
### Anomaly Detection
- Detect anomalies based on statistical thresholds.
- Flag transactions with amounts that significantly deviate from the mean.
### Reporting
- Generate a detailed report listing all detected anomalies, including transaction details and reasons for flagging.

## Workflow
### Load and Preprocess Data:
- Read the dataset and clean it.
### Calculate Statistics:
- Compute mean and standard deviation for each transaction type.
### Detect Anomalies:
- Identify transactions that fall outside the calculated thresholds.
### Generate Report:
- Create a CSV report with all detected anomalies.

## Conclusion
This project provides a complete solution for detecting anomalies in financial transactions. The approach combines data preprocessing, statistical analysis, and anomaly detection to identify unusual transaction patterns effectively.

## References
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [NumPy Documentation](https://numpy.org/doc/)

## Example Code
### Example Code
```python
import pandas as pd
import numpy as np

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces
    return df

# Calculate statistics for anomaly detection
def calculate_statistics(df):
    stats = df.groupby('type')['amount'].agg(['mean', 'std'])
    stats['z_score_threshold'] = 2.5  # Using a Z-score threshold of 2.5 for anomaly detection
    stats['upper_limit'] = stats['mean'] + stats['z_score_threshold'] * stats['std']
    stats['lower_limit'] = stats['mean'] - stats['z_score_threshold'] * stats['std']
    return stats

# Detect anomalies based on statistical thresholds
def detect_anomalies(df, stats):
    anomalies = []
    for _, row in df.iterrows():
        upper_limit = stats.loc[row['type'], 'upper_limit']
        lower_limit = stats.loc[row['type'], 'lower_limit']
        if row['amount'] > upper_limit or row['amount'] < lower_limit:
            reason = f"Anomaly detected: Amount {row['amount']} is outside the range ({lower_limit}, {upper_limit})"
            anomalies.append((row['step'], row['type'], row['amount'], row['nameOrig'], row['nameDest'], reason))
    return anomalies

# Generate anomaly report
def generate_anomaly_report(anomalies, output_file):
    anomaly_df = pd.DataFrame(anomalies, columns=['step', 'type', 'amount', 'nameOrig', 'nameDest', 'reason_for_anomaly'])
    anomaly_df.to_csv(output_file, index=False)
    print("Anomaly report generated successfully.")

# Main function to orchestrate the process
def main(file_path, output_file):
    df = load_and_preprocess_data(file_path)
    stats = calculate_statistics(df)
    anomalies = detect_anomalies(df, stats)
    generate_anomaly_report(anomalies, output_file)

# Adjust the file_path according to the location of your CSV file
file_path = r'C:\Users\aayus\OneDrive\Desktop\FIN TRANS\financial_transactions.csv'
output_file = r'C:\Users\aayus\OneDrive\Desktop\FIN TRANS\anomaly_report.csv'

main(file_path, output_file)

# Load the anomaly report and display it
anomaly_report_path = r'C:\Users\aayus\OneDrive\Desktop\FIN TRANS\anomaly_report.csv'
anomaly_df = pd.read_csv(anomaly_report_path)

# Display the anomaly report if there are anomalies detected
if not anomaly_df.empty:
    print(anomaly_df)
else:
    print("No anomalies detected based on the current criteria.")
