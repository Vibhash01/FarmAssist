from cropmatch import CropMatch
from agrofit import AgroFit
from yield_predictor import YieldPredictor
import warnings


def main():
    warnings.filterwarnings("ignore", message=".*libiomp.*libomp.*")
    data_path = "C:/Users/Atharva/Desktop/Agrofit/enlarged_agriculture_dataset.csv"

    # Initialize and save all models
    crop_match = CropMatch(data_path=data_path, model_path='crop_match_model.pkl')
    crop_match.save_model('crop_match_model.pkl')

    agrofit = AgroFit(data_path=data_path, model_path='agrofit_model.pkl')
    agrofit.save_model('agrofit_model.pkl')

    yield_predictor = YieldPredictor(data_path=data_path, model_path='yield_predictor_model.pkl')
    yield_predictor.save_model('yield_predictor_model.pkl')

    print("All models loaded and saved successfully.\n")

    print("Options:")
    print("[1] Get Crop Subtype Recommendation")
    print("[2] Get Recommended Conditions for Subtype and Variety")
    print("[3] Predict Yield for Your Crop")
    print("[4] Exit")
    print("Type '99' to see the options again.\n")

    
    while True:
        choice = input("Choose an option: ").strip()

        if choice == "1":
            print("\nLet's get a crop subtype recommendation.")
            N = float(input("Enter Nitrogen (N): "))
            P = float(input("Enter Phosphorus (P): "))
            K = float(input("Enter Potassium (K): "))
            temp = float(input("Enter Temperature (°C): "))
            humidity = float(input("Enter Humidity (%): "))
            pH = float(input("Enter pH: "))
            rainfall = float(input("Enter Rainfall (mm): "))
            recommendation = agrofit.recommend_subtype([N, P, K, temp, humidity, pH, rainfall])
            print(f"Recommended Crop Subtype: {recommendation}")

        elif choice == "2":
            subtype = input("Enter Crop Subtype: ")
            variety = input("Enter Crop Variety: ")
            conditions = agrofit.recommend_conditions(subtype, variety)
            print(f"\nRecommended Conditions for {subtype} ({variety}):")
            if isinstance(conditions, str):
                print(conditions)
            else:
                for key, value in conditions.items():
                    print(f"{key}: {value}")

        elif choice == "3":
            print("\nLet's predict the yield for your crop.")
            print("Please enter the following numerical features:")
            pH = float(input("pH Level: "))
            nitrogen = float(input("Nitrogen Content (ppm): "))
            phosphorus = float(input("Phosphorus Content (ppm): "))
            potassium = float(input("Potassium Content (ppm): "))
            rainfall = float(input("Rainfall (mm): "))
            temperature = float(input("Temperature (°C): "))
            humidity = float(input("Humidity (%): "))
            sunlight_hours = float(input("Sunlight Hours (per day): "))
            altitude = float(input("Altitude (m): "))
            growing_period = float(input("Growing Period (days): "))

            print("\nNow, please enter the following categorical features:")
            print("Soil Type options: ['Clay', 'Loam', 'Peat', 'Sandy', 'Silt']")
            soil_type = input("Soil Type: ")

            print("Planting Season options: ['Autumn', 'Spring', 'Summer', 'Winter']")
            planting_season = input("Planting Season: ")

            print("Harvesting Season options: ['Autumn', 'Spring', 'Summer', 'Winter']")
            harvesting_season = input("Harvesting Season: ")

            print("Subtype options: ['Apple', 'Banana', 'Barley', 'Maize', 'Onion', 'Potato', 'Rice', 'Tomato', 'Wheat']")
            subtype = input("Subtype: ")

            print("Crop Type options: ['Fruit', 'Grain', 'Vegetable']")
            crop_type = input("Crop Type: ")

            print("Varieties options:")
            print("['Basmati', 'Beefsteak', 'Cavendish', 'Cherry', 'Dent', 'Durum', 'Emmer', 'Fingerling', 'Flint', 'Fuji', 'Gala', 'Heirloom',")
            print("'Honeycrisp', 'Hulled', 'Jasmine', 'Khorasan', 'Lady Finger', 'Malting Barley', 'Pearl', 'Ponni', 'Popcorn', 'Red', 'Red Banana',")
            print("'Roma', 'Russet', 'Spelt', 'Sweet Corn', 'White', 'Wild Rice', 'Yellow', 'Yukon Gold']")
            variety = input("Variety: ")

            user_input = {
                "pH Level": pH,
                "Nitrogen Content (ppm)": nitrogen,
                "Phosphorus Content (ppm)": phosphorus,
                "Potassium Content (ppm)": potassium,
                "Rainfall (mm)": rainfall,
                "Temperature (°C)": temperature,
                "Humidity (%)": humidity,
                "Sunlight Hours (per day)": sunlight_hours,
                "Altitude (m)": altitude,
                "Growing Period (days)": growing_period,
                "Soil Type": soil_type,
                "Planting Season": planting_season,
                "Harvesting Season": harvesting_season,
                "Subtype": subtype,
                "Crop Type": crop_type,
                "Varieties": variety
            }

            yield_result = yield_predictor.predict_yield(user_input)
            if yield_result is not None:
                print(f"\nThe predicted yield is: {yield_result:.2f} kg/ha")
            else:
                print("Error: Could not predict yield. Please check your inputs.")

        elif choice == "4":
            print("Exiting application. Goodbye!")
            break

        elif choice == "99":
            print("\nOptions:")
            print("[1] Get Crop Subtype Recommendation")
            print("[2] Get Recommended Conditions for Subtype and Variety")
            print("[3] Predict Yield for Your Crop")
            print("[4] Exit")

        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
