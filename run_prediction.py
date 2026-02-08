from train_model import PurchaseAmountPredictor
import sys

def main():
    # Get filename from user
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = input("Enter the CSV filename (e.g., customer_purchases.csv): ").strip()
    
    # Create and train model
    predictor = PurchaseAmountPredictor()
    
    try:
        predictor.load_and_train(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found!")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*50)
    print("MODEL READY FOR PREDICTIONS")
    print("="*50)
    
    # Interactive prediction loop
    while True:
        print("\n" + "="*50)
        print("PREDICT A CUSTOMER'S PURCHASE AMOUNT")
        print("="*50)
        
        try:
            # Collect customer data
            customer_data = {}
            
            # Quantitative inputs
            customer_data['Age'] = int(input("Age: "))
            customer_data['Frequency_of_Purchase'] = int(input("Frequency of Purchase: "))
            customer_data['Product_Rating'] = int(input("Product Rating (1-5): "))
            customer_data['Customer_Satisfaction'] = int(input("Customer Satisfaction (1-10): "))
            
            # Categorical inputs
            customer_data['Purchase_Category'] = input("Purchase Category: ").strip()
            
            
            predictor.predict_purchase_amount(customer_data)
            
        except ValueError as e:
            print(f"Invalid input: {e}")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        
        # Ask if user wants to continue
        continue_input = input("\nPredict another? (y/n): ").strip().lower()
        if continue_input != 'y':
            break
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
