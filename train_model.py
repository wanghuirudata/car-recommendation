import pandas as pd
from model.regressor import FEATURE_COLUMNS, Model

def main():
    try:
        # Load your data
        data = pd.read_csv('vehicle.csv')  # Adjust path as needed
        
        # Train and save the model
        model = Model.train_and_save_model(data)
        print("Model trained and saved successfully!")
        
        # Optional: Test the model
        test_data = data.iloc[0:1][FEATURE_COLUMNS]
        prediction = model.predict(test_data)
        print(f"Test prediction: {prediction[0]}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")

if __name__ == "__main__":
    main()