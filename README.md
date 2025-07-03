## Machine Learning Project: Time Series Forecasting with LSTM

### Overview
This project forecasts stock prices (e.g., NKE) using an LSTM neural network in PyTorch. The workflow covers data preparation, feature engineering, scaling, model training, hyperparameter tuning, evaluation, and visualization.

---

### Workflow Steps

1. **Data Preparation & Visualization**
   - Loads historical stock price data.
   - Keeps only the 'Close' price and ensures 'Date' is a column.
   - Plots the closing price over time.

2. **Feature Engineering**
   - Implements a lagging function to create lookback features (e.g., `Close(t-1)` to `Close(t-7)`).
   - Drops rows with NaN values from shifting.

3. **Data Scaling**
   - Converts the DataFrame to a NumPy array.
   - Scales features to [-1, 1] using `MinMaxScaler`.

4. **Train/Test Split**
   - Splits data into features (X) and target (y).
   - Uses 95% for training, 5% for testing.

5. **Reshaping & Tensor Conversion**
   - Reshapes X for LSTM input: (samples, lookback, 1).
   - Converts X and y to PyTorch tensors.

6. **Dataset & DataLoader**
   - Defines a custom `TimeSeriesDataset` for PyTorch.
   - Creates DataLoaders for training and testing (batch size 16).

7. **Model Definition**
   - Defines an LSTM model class with configurable hyperparameters (input size, hidden size, layers, dropout).
   - Uses LSTM, dropout, and a linear output layer.

8. **Training & Validation Functions**
   - Implements functions for training and validating one epoch.
   - Prints loss statistics.

9. **Hyperparameter Grid Search**
   - Searches over hidden size, number of layers, learning rate, and dropout.
   - Trains each combination for 5 epochs.
   - Selects the best parameters based on validation loss.

10. **Final Model Training**
    - Trains the best model for 10 epochs.
    - Prints training loss per epoch.

11. **Evaluation**
    - Predicts on the test set.
    - Inverse-transforms predictions to original scale.
    - Calculates and prints test RMSE (e.g., `Final Test RMSE: 2.2498`).

12. **Plotting Results**
    - Plots predicted vs. actual closing prices for the test set.

---

### Libraries Used
- PyTorch
- pandas
- numpy
- matplotlib
- scikit-learn

---

### Key Hyperparameters (Best Found)
- `hidden_size=8`
- `num_stacked_layers=1`
- `learning_rate=0.01`
- `dropout_prob=0.0`
