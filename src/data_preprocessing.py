
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Handle 'Gender' column
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # Standardize numerical features
    cols_to_scale = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    return df, scaler



