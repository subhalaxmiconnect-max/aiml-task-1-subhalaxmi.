def main():
  print("[INFO] Starting AIML Task - 1: House Price Prediction Model...\n")

  # 1. Generate or Load Dataset
  filepath = "dataset.csv"
  np.random.seed(42)
  n_samples = 500

  square_feet = np.random.randint(800, 3500, size=n_samples)
  bedrooms = np.random.randint(1, 5, size=n_samples)
  bathrooms = np.random.randint(1, 4, size=n_samples)
  age = np.random.randint(0, 30, size=n_samples)

  price = (
      (square_feet * 150)
      + (bedrooms * 10000)
      + (bathrooms * 15000)
      - (age * 1000)
      + np.random.normal(0, 20000, size=n_samples)
  )

  df = pd.DataFrame({
      "SquareFeet": square_feet,
      "Bedrooms": bedrooms,
      "Bathrooms": bathrooms,
      "Age": age,
      "Price": price,
  })
  df.to_csv(filepath, index=False)
  print(f"[INFO] Dataset loaded successfully from {filepath}")

  # 2. Define Features and Target
  X = df[["SquareFeet", "Bedrooms", "Bathrooms", "Age"]]
  y = df["Price"]

  # 3. Train-Test Split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  # 4. Feature Scaling
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  # 5. Model Training
  print("[INFO] Training Random Forest Regressor model...")
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train_scaled, y_train)

  # 6. Evaluation
  y_pred = model.predict(X_test_scaled)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  r2 = r2_score(y_test, y_pred)

  print("\n--- Model Evaluation Results ---")
  print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
  print(f"R² Score: {r2:.4f}")

  # 7. Sample Prediction
  sample_input = np.array([[2000, 3, 2, 5]])
  sample_scaled = scaler.transform(sample_input)
  predicted_price = model.predict(sample_scaled)[0]

  print("\n--- Sample Prediction Test ---")
  print(
      "Input Properties: 2000 sqft | 3 Bedrooms | 2 Bathrooms | 5 Years Old"
  )
  print(f"Predicted House Price: ${predicted_price:,.2f}")


if __name__ == "__main__":
  main()
  
