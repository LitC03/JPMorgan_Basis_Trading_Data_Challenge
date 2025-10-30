import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import math


def load_data(filename):
    raw_data = pd.read_csv(filename)
    # Give name to dates
    raw_data = raw_data.rename(columns={'Unnamed: 0':'Dates'})
    # Load datasets
    dataset1 = (raw_data["German Apples (EUR)"]*raw_data["EURGBP"]).to_numpy()
    dataset2 = raw_data["UK Apples (GBP)"].to_numpy()

    # Obtain the basis
    X =  dataset2 - dataset1 
    m = X.mean()
    sigma = X.std()
    X = (X -m) / sigma # normalise data
    N = len(raw_data)

    # Obtain dates array
    time_arr = pd.to_datetime(raw_data["Dates"],format='%d/%m/%Y')

    return X,N,m,sigma,dataset2,dataset1,time_arr

def get_OU_params(X,dt):

    # Creating the y-vector of the regression (X at time t+1)
    y = X[1:]

    # Creating the x-vector (X at time t)
    X_t = X[:-1].reshape(-1, 1)

    # Fit the linear model: y = a + b*X_t
    model = LinearRegression()
    model.fit(X_t, y)

    # Get the regression parameters
    a = model.intercept_
    b = model.coef_[0]

    # Calculating the residuals and their standard deviation
    residuals = y - model.predict(X_t)
    std_residuals = np.std(residuals)

    # Get the parameters for a Ornstein–Uhlenbeck process

    theta_fit = -np.log(b) / dt
    mu_fit = a / (1 - b)
    sigma_fit = std_residuals * np.sqrt( (2 * theta_fit) / (1 - b**2) )
    
    print("--- Fitted Ornstein-Uhlenbeck Parameters ---")
    print(f"Reversion Speed (theta): {theta_fit:.4f}")
    print(f"Long-Term Mean (mu):     {mu_fit:.4f}")
    print(f"Volatility (sigma):    {sigma_fit:.4f}")

    return theta_fit,mu_fit,sigma_fit


def simulate_results(params,basis,N,dt,UK_dataset,German_dataset,mean,sigma,time_arr):

    X_simulated = np.zeros(N)

    # Start the simulation from the same first point as the actual data
    X_simulated[0] = basis[0]

    # Loop through and generate the data according to the known basis
    for t in range(1, N):
        X_simulated[t] = get_new_step(params,basis[t-1],dt)

         
    # 'Unnormalise' the basis to return to GPB values
    X_unnorm = (X_simulated*sigma)+mean

     # Get estimate of UK prices
    estimated_apple_prices = German_dataset+X_unnorm

    # Calculate RMSE
    RMSE = math.sqrt(math.fsum((UK_dataset - estimated_apple_prices)**2)/N)
    print(f'RMSE of model: {RMSE}')

    # plot_everything(estimated_apple_prices,UK_dataset,German_dataset,time_arr,N)
    return RMSE

def get_new_step(params,previous_basis,dt):
    # Get parameters
    theta_fit,mu_fit,sigma_fit = params

    # Get the random shock (Wiener process)
    dW = np.random.normal(loc=0.0, scale=np.sqrt(dt))

    # Apply the discretized OU equation (Euler-Maruyama)
    new_step = (
            previous_basis + 
            theta_fit * (mu_fit - previous_basis) * dt + 
            sigma_fit * dW
        )
    
    return new_step

def estimate_new_price(params,curr_price_german,previous_basis,dt,mean,sigma):
    new_step = get_new_step(params,previous_basis,dt)
    new_price = curr_price_german + ((new_step*sigma)+mean)
    return new_price

def plot_everything(estimated_price,uk,german,time_arr,N):

    plt.figure(figsize=(12, 7))
    plt.plot(time_arr,uk,label='UK apple prices (GBP)', color='blue', alpha=0.9, linewidth=1.5)
    plt.plot(time_arr,estimated_price,label='German apple prices + Simulated basis (GBP)', color='red', alpha=0.9, linewidth=1)
    plt.title('UK apple prices (GBP) vs. Estimated UK apple prices from German market ', fontsize=16)
    plt.xlabel('Time Step')
    plt.ylabel('Apple prices (GBP)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


def main():
    basis,N,mean,sigma,UK_dataset,German_dataset,time_arr = load_data("apples_exercise.csv")
    dt = 1/252 # obtain dt value for daily data
    params = get_OU_params(basis,dt)

    reps = 100
    RMSE = np.zeros((reps,))
    print(len(RMSE))
    for i in range(reps):
        RMSE[i] = simulate_results(params,basis,N,dt,UK_dataset,German_dataset,mean,sigma,time_arr)
    
    mean_rmse = RMSE.mean()
    std_rmse = RMSE.std()

    print(f"Mean RMSE: {mean_rmse:.3f}")
    print(f"Std RMSE: {std_rmse:.3f}")
    # Testing
    t = 10
    estimated_price = estimate_new_price(params,German_dataset[t],basis[t-1],dt,mean,sigma)
    print(f'At t={t}, the estimate for UK apples is {estimated_price:.2f} GBP')
    print(f'The actual price was {UK_dataset[t]:.2f} GBP')

if __name__ == '__main__':
    main()