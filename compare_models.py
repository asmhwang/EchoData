from train_model import PurchaseAmountPredictor
import sys
import pandas as pd

def main():
    print("="*60)
    print("COMPARE TWO DATASETS")
    print("="*60)
    
    # Get filenames from user
    if len(sys.argv) >= 3:
        filename1 = sys.argv[1]
        filename2 = sys.argv[2]
    else:
        filename1 = input("Enter first CSV filename: ").strip()
        filename2 = input("Enter second CSV filename: ").strip()
    
    # Train on first dataset
    print("\n" + "="*60)
    print(f"TRAINING MODEL 1 ON: {filename1}")
    print("="*60)
    predictor1 = PurchaseAmountPredictor()
    
    try:
        predictor1.load_and_train(filename1)
    except FileNotFoundError:
        print(f"Error: File '{filename1}' not found!")
        return
    except Exception as e:
        print(f"Error loading first dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Train on second dataset
    print("\n" + "="*60)
    print(f"TRAINING MODEL 2 ON: {filename2}")
    print("="*60)
    predictor2 = PurchaseAmountPredictor()
    
    try:
        predictor2.load_and_train(filename2)
    except FileNotFoundError:
        print(f"Error: File '{filename2}' not found!")
        return
    except Exception as e:
        print(f"Error loading second dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compare the models
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    # Get errors from both models
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    import numpy as np
    
    # Re-evaluate model 1
    X1 = predictor1.df[predictor1.feature_columns]
    y1 = predictor1.df['Purchase_Amount']
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    y1_pred = predictor1.model.predict(X1_test)
    error1 = mean_absolute_error(y1_test, y1_pred)
    
    # Re-evaluate model 2
    X2 = predictor2.df[predictor2.feature_columns]
    y2 = predictor2.df['Purchase_Amount']
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
    y2_pred = predictor2.model.predict(X2_test)
    error2 = mean_absolute_error(y2_test, y2_pred)
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Metric': [
            'Dataset',
            'Number of Samples',
            'Average Error',
            'Purchase Amount Range',
            'Most Important Feature'
        ],
        'Model 1': [
            filename1,
            len(predictor1.df),
            f"${error1:.2f}",
            f"${predictor1.df['Purchase_Amount'].min():.2f} to ${predictor1.df['Purchase_Amount'].max():.2f}",
            predictor1.feature_columns[np.argmax(predictor1.model.feature_importances_)]
        ],
        'Model 2': [
            filename2,
            len(predictor2.df),
            f"${error2:.2f}",
            f"${predictor2.df['Purchase_Amount'].min():.2f} to ${predictor2.df['Purchase_Amount'].max():.2f}",
            predictor2.feature_columns[np.argmax(predictor2.model.feature_importances_)]
        ]
    })
    
    print("\n", comparison_df.to_string(index=False))
    
    # Show side-by-side sample predictions
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS COMPARISON (First 10 test samples)")
    print("="*60)
    
    # Get same number of samples from both (use minimum)
    num_samples = min(10, len(y1_test), len(y2_test))
    
    comparison_samples = pd.DataFrame({
        'Model 1 Actual': ['$' + f"{val:.2f}" for val in y1_test.values[:num_samples]],
        'Model 1 Predicted': ['$' + f"{val:.2f}" for val in y1_pred[:num_samples]],
        'Model 1 Error': ['$' + f"{val:.2f}" for val in (y1_test.values[:num_samples] - y1_pred[:num_samples])],
        'Model 2 Actual': ['$' + f"{val:.2f}" for val in y2_test.values[:num_samples]],
        'Model 2 Predicted': ['$' + f"{val:.2f}" for val in y2_pred[:num_samples]],
        'Model 2 Error': ['$' + f"{val:.2f}" for val in (y2_test.values[:num_samples] - y2_pred[:num_samples])]
    })
    
    print("\n", comparison_samples.to_string(index=True))
    
    # Determine winner
    print("\n" + "="*60)
    if error1 < error2:
        print(f"  MODEL 1 WINS! (Lower error: ${error1:.2f} vs ${error2:.2f})")
        print(f"   Model 1 is {((error2 - error1) / error2 * 100):.1f}% more accurate")
    elif error2 < error1:
        print(f"  MODEL 2 WINS! (Lower error: ${error2:.2f} vs ${error1:.2f})")
        print(f"   Model 2 is {((error1 - error2) / error1 * 100):.1f}% more accurate")
    else:
        print(f"  TIE! Both models have the same error: ${error1:.2f}")
    print("="*60)
    
    # Test on same inputs
    print("\n" + "="*60)
    print("INTERACTIVE SIDE-BY-SIDE PREDICTION TEST")
    print("="*60)
    
    while True:
        try:
            print("\nEnter customer details to compare predictions:")
            
            # Collect customer data
            customer_data = {}
            
            # Quantitative inputs
            customer_data['Age'] = int(input("Age: "))
            customer_data['Frequency_of_Purchase'] = int(input("Frequency of Purchase: "))
            customer_data['Product_Rating'] = int(input("Product Rating (1-5): "))
            customer_data['Customer_Satisfaction'] = int(input("Customer Satisfaction (1-10): "))
            
            # Categorical inputs (simplified - ask for key ones)
            customer_data['Purchase_Category'] = input("Purchase Category: ").strip()
            customer_data['Device_Used_for_Shopping'] = input("Device Used (Mobile/Desktop/Tablet): ").strip()
            
            
            
            # Get predictions from both models
            print("\n" + "-"*60)
            print("Getting predictions from both models...")
            print("-"*60)
            
            pred1 = predictor1.predict_purchase_amount(customer_data.copy())
            pred2 = predictor2.predict_purchase_amount(customer_data.copy())
            
            # Display side by side
            print("\n" + "-"*60)
            print(f"{'Model 1 Prediction:':<30} {'Model 2 Prediction:':<30}")
            print(f"${pred1:<29.2f} ${pred2:<29.2f}")
            print(f"\nDifference: ${abs(pred1 - pred2):.2f}")
            print("-"*60)
            
        except ValueError as e:
            print(f"Invalid input: {e}")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        
        continue_input = input("\nCompare another prediction? (y/n): ").strip().lower()
        if continue_input != 'y':
            break
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
