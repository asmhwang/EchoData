import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import warnings

class PurchaseAmountPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.categorical_columns = [
            'Purchase_Category',
        ]
        self.quantitative_columns = [
            'Age',
            'Frequency_of_Purchase',
            'Product_Rating',
            'Customer_Satisfaction'
        ]
        self.feature_columns = None
        self.df = None
        
    def clean_data(self, df):
        """Clean and convert data types"""
        # Make a copy to avoid warnings
        df = df.copy()
        
        # Clean Purchase_Amount (remove $ and commas)
        if 'Purchase_Amount' in df.columns:
            df['Purchase_Amount'] = df['Purchase_Amount'].astype(str).str.replace('$', '', regex=False)
            df['Purchase_Amount'] = df['Purchase_Amount'].str.replace(',', '', regex=False)
            df['Purchase_Amount'] = df['Purchase_Amount'].str.strip()
            df['Purchase_Amount'] = pd.to_numeric(df['Purchase_Amount'], errors='coerce')
        
        # Clean quantitative columns
        for col in self.quantitative_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('$', '', regex=False)
                df[col] = df[col].str.replace(',', '', regex=False)
                df[col] = df[col].str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing Purchase_Amount (target variable)
        df = df.dropna(subset=['Purchase_Amount'])
        
        # Fill missing values in quantitative columns with median
        for col in self.quantitative_columns:
            if col in df.columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Fill missing values in categorical columns with mode
        for col in self.categorical_columns:
            if col in df.columns:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col] = df[col].fillna(mode_val)
        
        return df
        
    def load_and_train(self, filename):
        """Load data from CSV and train the model"""
        print(f"Loading data from {filename}...")
        df_raw = pd.read_csv(filename)
        
        print(f"Original dataset shape: {df_raw.shape}")
        
        # Clean the data
        print("Cleaning data...")
        self.df = self.clean_data(df_raw)
        
        print(f"After cleaning: {self.df.shape}")
        
        # Verify Purchase_Amount is numeric
        if self.df['Purchase_Amount'].dtype not in ['float64', 'int64']:
            print(f"Error: Purchase_Amount is still type {self.df['Purchase_Amount'].dtype}")
            print("Sample values:", self.df['Purchase_Amount'].head())
            return None
        
        # Check the data
        print(f"\nFirst few rows:")
        print(self.df.head())
        print(f"\nPurchase Amount - Min: ${self.df['Purchase_Amount'].min():.2f}, Max: ${self.df['Purchase_Amount'].max():.2f}, Mean: ${self.df['Purchase_Amount'].mean():.2f}")
        
        # Encode categorical variables
        for col in self.categorical_columns:
            if col in self.df.columns:
                self.label_encoders[col] = LabelEncoder()
                self.df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(self.df[col].astype(str))
        
        # Build feature column list
        encoded_categorical = [f'{col}_encoded' for col in self.categorical_columns if col in self.df.columns]
        available_quantitative = [col for col in self.quantitative_columns if col in self.df.columns]

        self.feature_columns = available_quantitative + encoded_categorical
        
        
        print(f"\nUsing {len(self.feature_columns)} features:")
        print(f"  - {len(available_quantitative)} quantitative")
        print(f"  - {len(encoded_categorical)} categorical (encoded)")
        
        # Define features (X) and target (y)
        X = self.df[self.feature_columns].copy()
        y = self.df['Purchase_Amount'].copy()
        
        # Check for any remaining NaN values
        if X.isnull().any().any():
            print("\nWarning: Found NaN values in features, filling with 0...")
            X = X.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest Regressor
        print("\nTraining Random Forest Regressor...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        # Calculate average error
        avg_error = mean_absolute_error(y_test, y_pred)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        print(f"Average Error: ${avg_error:.2f}")
        
        # Show some example predictions
        print("\nSample Predictions vs Actual:")
        comparison = pd.DataFrame({
            'Actual': ['$' + f"{val:.2f}" for val in y_test.values[:10]],
            'Predicted': ['$' + f"{val:.2f}" for val in y_pred[:10]],
            'Error': ['$' + f"{val:.2f}" for val in (y_test.values[:10] - y_pred[:10])]
        })
        print(comparison)
        
        # Feature importance (top 10)
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Visualize predictions vs actual
        
        return self
    
    def predict_purchase_amount(self, customer_data):
        """
        Predict the purchase amount for a customer
        
        Parameters:
        - customer_data: dict with all required features
        """
        if self.model is None:
            print("Error: Model not trained yet!")
            return None
        
        # Create a copy to avoid modifying original
        data = customer_data.copy()
        
        
        # Encode categorical variables
        for col in self.categorical_columns:
            if col in data:
                # Handle unknown categories
                if str(data[col]) not in self.label_encoders[col].classes_:
                    print(f"Warning: '{data[col]}' not seen in training data for {col}. Using most common value.")
                    data[col] = self.df[col].mode()[0]
                data[f'{col}_encoded'] = self.label_encoders[col].transform([str(data[col])])[0]
        
        # Build feature dictionary in correct order
        feature_dict = {}
        for col in self.feature_columns:
            if col.endswith('_encoded'):
                feature_dict[col] = [data[col]]
            elif col in data:
                feature_dict[col] = [data[col]]
            else:
                print(f"Warning: Missing feature {col}, using 0")
                feature_dict[col] = [0]
        
        # Create DataFrame
        new_data = pd.DataFrame(feature_dict)
        
        prediction = self.model.predict(new_data)[0]
        
        print(f"\nPredicted Purchase Amount: ${prediction:.2f}")
        
        return prediction
    
    def get_feature_summary(self):
        """Return summary of features used"""
        if self.df is None:
            return "Model not trained yet"
        
        return {
            'categorical_features': self.categorical_columns,
            'quantitative_features': self.quantitative_columns,
            'total_features': len(self.feature_columns)
        }
