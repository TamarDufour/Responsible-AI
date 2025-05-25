import pandas as pd
import matplotlib.pyplot as plt

def confusion_matrix_to_df(loaded_model, X_test, y_test):
    """
    Converts a confusion matrix to a DataFrame.

    Args:
        confusion_matrix (array-like): The confusion matrix to convert.

    Returns:
        pd.DataFrame: A DataFrame representation of the confusion matrix.
    """
    from sklearn.metrics import confusion_matrix

    y_pred = loaded_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Create a DataFrame from the confusion matrix
    cm_df = pd.DataFrame(cm,
                         index=loaded_model.classes_,
                         columns=loaded_model.classes_)

    return cm_df

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


def patient_color():
    pass


if __name__ == "__main__":
    metadata = pd.read_csv(r"C:\Users\admin\PycharmProjects\Responsible-AI\data\HAM10000_metadata.csv")
    metadata = metadata.dropna(subset=['dx'])
    print(metadata.head())
    #sample randomly only 50% of the nv sample, to undersample this group
    nv_sample = metadata[metadata['dx'] == 'nv']
    nv_sample = nv_sample.sample(frac=0.4, random_state=42)
    metadata_not_nv = metadata[metadata['dx'] != 'nv']
    undersampled_metadata = pd.concat([nv_sample, metadata_not_nv])

    #union the dx category of bkl, bcc, akiec, vasc, df to NMSC
    undersampled_metadata['dx'] = undersampled_metadata['dx'].replace({
        'bkl': 'NMSC',
        'bcc': 'NMSC',
        'akiec': 'NMSC',
        'vasc': 'NMSC',
        'df': 'NMSC'
    })

    import torch

    print("CUDA available:", torch.cuda.is_available())
    print("GPU device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

    EDA(undersampled_metadata)






