import numpy as np
from scipy.linalg import solve_triangular, solve


class SimpleLinearRegression:
    """
    Implement simple linear regression using numpy  
    """
    def __init__(self, responses: np.ndarray, predictors: np.ndarray):
        """
        Parameters
        ---------
        responses: np.ndarray
            The variable to predict
        
        predictors: np.ndarray
            The independent variables using which predictions will be made.
        """
        assert type(responses) == np.ndarray
        assert type(predictors) == np.ndarray
        assert len(responses) == len(predictors)
        assert len(responses.shape) == 1
        self.y = responses
        self.x = predictors
        self.num_of_points = len(self.y)
    
    def fit(self):
        u = np.ones(self.x.shape)
        alpha_num = self.x.T.dot(self.y) - (u.T.dot(self.x) * u.T.dot(self.y)) / self.num_of_points
        alpha_denom = (self.x.T.dot(self.x) - (u.T.dot(self.x)**2) / self.num_of_points)
        alpha = alpha_num / alpha_denom
        beta = u.T.dot((self.y - (alpha * self.x))) / self.num_of_points
        self.alpha = alpha
        self.beta = beta
        print(f"alpha: {alpha}")
        print(f"beta: {beta}")
        return (alpha, beta)

    def predict(self, x: np.ndarray):
        assert type(x) == np.ndarray
        y_predicted = self.alpha * x + self.beta
        return y_predicted


class MultipleLinearRegressionNormalMethod:
    """
    Implements normal method to create a best-fit for the data
    """
    def __init__(self, x : np.ndarray, y: np.ndarray):
        assert type(x) == np.ndarray
        assert type(y) == np.ndarray
        assert len(x) == len(y)
        self.x = x
        self.y = y
    
    def fit(self):
        C = self.x.T.dot(self.x)
        # theta = np.dot(np.linalg.inv(C),self.x.T.dot(self.y))
        b = self.x.T.dot(self.y)
        theta = solve(C,b)
        self.theta = theta
    
    def predict(self, x_test: np.ndarray):
        return x_test.dot(self.theta)
    
class MultipleLinearRegressionQRMethod:
    """
    Implements QR decomposition method to create a best-fit for the data
    """
    def __init__(self, x : np.ndarray, y: np.ndarray):
        assert type(x) == np.ndarray
        assert type(y) == np.ndarray
        assert len(x) == len(y)
        self.x = x
        self.y = y
    
    def fit(self):
        Q, R = np.linalg.qr(self.x)
        z = Q.T.dot(self.y)
        self.theta = solve_triangular(R,z)
    
    def predict(self, x_test: np.ndarray):
        return x_test.dot(self.theta)