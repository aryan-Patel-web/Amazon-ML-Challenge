import os
import pandas as pd
import lightgbm as lgb  # Make sure you have your trained model loaded
import joblib           # If you saved your model with joblib

# Load your trained model (adjust filename if needed)
final_model = joblib.load(os.path.join('dataset/', 'lgbm_model.pkl'))

def predictor(sample_id, catalog_content, image_link, df_features):
    '''
    Predict price using your trained LightGBM model
    
    Parameters:
    - sample_id: Unique identifier for the sample
    - catalog_content: Text containing product title and description (not used directly here)
    - image_link: URL to product image (not used directly here)
    - df_features: DataFrame containing features ready for prediction
    
    Returns:
    - price: Predicted price as float
    '''
    # Extract the row corresponding to this sample_id
    row = df_features[df_features['sample_id'] == sample_id].drop(columns=['sample_id'])
    
    # Predict using your trained model
    price = final_model.predict(row)[0]
    
    return float(price)

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    
    # Read test data
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    # Load preprocessed feature CSV (sample test ready)
    # This should contain all features your model expects
    test_features = pd.read_csv(os.path.join(DATASET_FOLDER, 'test_ready.csv'))
    
    # Apply predictor function to each row
    test['price'] = test['sample_id'].apply(
        lambda sid: predictor(sid, None, None, test_features)
    )
    
    # Select only required columns for output
    output_df = test[['sample_id', 'price']]
    
    # Save predictions
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    output_df.to_csv(output_filename, index=False)
    
    print(f"Predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Sample predictions:\n{output_df.head()}")
