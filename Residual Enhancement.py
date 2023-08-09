"""
\item For each iteration \( t \) from 1 to \( T \) (where \( T \) is the number of trees):
\begin{enumerate}
    \item \textbf{Compute Pseudo-residuals:} Calculate the pseudo-residuals based on the negative gradient of the loss function concerning the model's predictions up to the previous iteration.
    \[
    r_{ti} = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} = y_i - F_{t-1}(x_i)
    \]
    Where:
    \begin{itemize}
        \item \( L(y_i, F(x_i)) \) is the loss function (e.g., MSE for regression).
        \item \( y_i \) is the actual value for the \( i^{th} \) instance.
        \item \( F_{t-1}(x_i) \) is the prediction of the \( i^{th} \) instance by the model up to iteration \( t-1 \).
    \end{itemize}
    
    \item \textbf{Train a Decision Tree:} Fit a decision tree, \( h_t(x) \), to the pseudo-residuals.
    
    \item \textbf{Update Model:} Update the model's prediction function with the newly trained decision tree.
    \[
    F_t(x) = F_{t-1}(x) + \alpha h_t(x)
    \]
    Where \( \alpha \) is the learning rate.
\end{enumerate}

This outline provides a clear structure for the Residual Enhancement method. 
Adjustments and optimizations can be made to this basic outline based on specific requirements or objectives.
"""
\

import numpy as np
from sklearn.tree import DecisionTreeRegressor

class SimpleRE:
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []

    def fit(self, X, y):
        # Initialize with the mean of y
        y_pred = np.full(np.shape(y), np.mean(y))
        
        for _ in range(self.n_estimators):
            # Compute the negative gradient (residuals)
            residuals = y - y_pred
            
            # Fit a decision tree to the residuals
            tree = DecisionTreeRegressor(max_depth=3)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Update predictions
            y_pred += self.learning_rate * tree.predict(X)

    def predict(self, X):
        # Initialize with the mean of the training data
        predictions = np.full(X.shape[0], np.mean(y))
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
            
        return predictions

# Test Sample usage: our team uses the top 50 errors; the Boston set is tested only
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load dataset
boston = load_boston()
X = boston.data
y = boston.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and predict
gbm = SimpleRE(n_estimators=100, learning_rate=0.1)
gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")
