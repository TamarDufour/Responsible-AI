import pandas as pd
import os
import matplotlib.pyplot as plt

DATA_DIR = "data" #CHANGE TO YOUR DATA DIRECTORY
META_FILE = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

def EDA(metadata):
    """
    Perform exploratory data analysis on the metadata DataFrame.

    Args:
        metadata (pd.DataFrame): The metadata DataFrame containing the dataset information.
    """
    print("Metadata Overview:")
    print(metadata.info())
    print("\nFirst few rows of metadata:")
    print(metadata.head())

    # Summary statistics
    print("\nSummary statistics of numerical columns:")
    print(metadata.describe())

    dx_counts = metadata['dx'].value_counts()
    dx_percentage = (dx_counts / dx_counts.sum()) * 100

    dx_summary = pd.DataFrame({
        'Count': dx_counts,
        'Percentage': dx_percentage
    })
    print("Diagnosis counts:")
    print(dx_summary)

    gender_counts = metadata['sex'].value_counts()
    gender_percentage = (gender_counts / gender_counts.sum()) * 100

    gender_summary = pd.DataFrame({'Count': gender_counts, 'Percentage:': gender_percentage})

    print("Gender counts:")
    print(gender_summary)

    # Plot diagnosis distribution
    plt.figure(figsize=(10, 6))
    dx_counts.plot(kind='bar')
    plt.title('Diagnosis Distribution in HAM10000')
    plt.xlabel('Diagnosis Type (dx)')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot as a bar chart
    plt.figure(figsize=(10, 6))
    dx_counts.plot(kind='bar')
    plt.title('Diagnosis Distribution in HAM10000')
    plt.xlabel('Diagnosis Type (dx)')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    metadata = pd.read_csv(META_FILE)
    print(metadata.head())
    print(metadata["dx_type"].value_counts())
    EDA(metadata)