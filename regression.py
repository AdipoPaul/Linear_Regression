import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (you'll need to provide the correct path)
data = pd.read_csv('Nairobi_Office_Price_Ex.csv')

# Feature and target variables
X = data['SIZE'].values
y = data['PRICE'].values

# Normalize features (Optional but often helpful)
X = (X - np.mean(X)) / np.std(X)


# Mean Squared Error function
def compute_mse(X, y, m, c):
    N = len(y)
    predictions = m * X + c
    mse = (1 / N) * np.sum((predictions - y) ** 2)
    return mse


# Gradient Descent function
def gradient_descent(X, y, m, c, learning_rate):
    N = len(y)
    predictions = m * X + c

    # Compute gradients
    dm = (-2 / N) * np.sum(X * (y - predictions))
    dc = (-2 / N) * np.sum(y - predictions)

    # Update weights
    m -= learning_rate * dm
    c -= learning_rate * dc

    return m, c


# Initialize parameters
m = np.random.rand()  # Random slope
c = np.random.rand()  # Random intercept
learning_rate = 0.1  # Reduce the learning rate
epochs = 10  # Increase the number of epochs

# Training the model
for epoch in range(epochs):
    mse = compute_mse(X, y, m, c)
    print(f"Epoch {epoch + 1}: MSE = {mse}")

    # Update m and c using gradient descent
    m, c = gradient_descent(X, y, m, c, learning_rate)

# Plot the line of best fit
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, m * X + c, color='red', label='Line of best fit')
plt.xlabel('Normalized Office Size (sq ft)')
plt.ylabel('Office Price')
plt.title('Linear Regression Line of Best Fit')
plt.legend()
plt.show()

# Prediction for office size of 100 sq. ft (unnormalized)
size = 100
normalized_size = (size - np.mean(data['SIZE'])) / np.std(data['SIZE'])
predicted_price = m * normalized_size + c
print(f"Predicted price for 100 sq. ft office: {predicted_price}")
