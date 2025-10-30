# Basis Trading Challenge
The objective of this analysis, set by JPMorgan, was to model the price basis between historical UK and German apple prices.

Based on the premise that this basis follows an Ornstein-Uhlenbeck (OU) process, this script estimates the process parameters with the use of an Autoregressive (AR) model. 
The fitted model is then used to predict UK apple prices based on the previous day's price basis and the concurrent German apple price (converted to GBP).

The model achieves a **Root Mean Square Error (RMSE) of 6.23 $\pm$ 0.08 GBP**, as illustrated in the figure below.

## Workflow
The script executes the following steps:
1. **Load data** (`load_data`):
   * Uses `pandas` to read `apples_exercise.csv`.
   * Converts the German apple price from EUR to GBP using the EURGBP exchange rate.
   * **Calculates the price basis** (spread) as: `X = UK_Apples_GBP - German_Apples_GBP`.
   * **Normalizes** the basis X by subtracting its mean and dividing by its standard deviation (z-score). This is done to stabilize the parameter fitting.
2. **Fit OU Parameters** (`get_OU_params`):
   * Models the normalized basis `X` as an **Ornstein-Uhlenbeck process**.
   * It uses Ordinary Least Squares (OLS) (via `LinearRegression`) to fit an AR(1) model:
     $X_{t+1} = a + b \cdot X_t$
     It then solves for the continuous-time OU parameters ($\theta, \mu, \sigma$) from the regression coefficients ($a, b$).
3. **Simulate UK market apple prices** (`simulate_results`):
   * Generates a new, full-length simulated path for the normalized basis using the fitted OU parameters.
   * Un-normalizes this simulated basis using the original mean and sigma **to get a simulated spread in GBP**.
   * Calculates the Estimated UK Price as: `Estimated_UK_Price = German_Apples_GBP + Simulated_Unnormalized_Basis`.
   * Calculates the RMSE between this `Estimated_UK_Price` and the actual UK_Apples_GBP data.
   * Returns this single RMSE value.
4. **Spot Test** (`main`):
   * Finally, the script runs a single one-step-ahead prediction test (`estimate_new_price`) for a specific day (t=10) to show a concrete example of the model in action.
