## Simple Linear regression

Formula for univariate model of linear regression is given by:  
$y \sim \alpha \cdot x + \beta$  

To fit a model values of $\alpha$ abd $\beta$ are as follows: 

$$
    \alpha  =  \dfrac{x^T y - \frac{1}{m}(u^T x)(u^T y)}
                     {x^T x - \frac{1}{m}(u^T x)^2} \\
    \beta  =  \frac{1}{m} u^T (y - \alpha x)
$$

where $ x $ is the predictor variables array, $ y $ is the responses array, $u$ is a vector of all ones and $m$ are the number of data points.


## Multiple Linear Regression


The data consists of $m$ observations and $n+1$ variables. One of these variables is the _response_ variable, $y$, which can be predicted from the other $n$ variables, $\{x_0, \ldots, x_{n-1}\}$. We want to fit a linear model which best fits the data. 

$$y_i \approx x_{i,0} \theta_0 + x_{i,1} \theta_1 + \cdots + x_{i,n-1} \theta_{n-1} + \theta_n,$$

where $\{\theta_j | 0 \leq j \leq n\}$ is the set of unknown coefficients. 
It can also be denoted as 

$$
  y \approx X \theta,
$$

where $X$ is the (input) data matrix.  

Our goal is to minimize the cost function(the sum of squared residuals)  
$$
    \|X \theta - y\|_2^2
$$

So we need to find $\theta^*$ where the cost function is minimal.  

There are two methods two solve this equation:  

### 1. Normal Equation Method

The solution derived from this method is 

$$
    X^TX \theta^* = X^Ty
$$

Therefore, 
$$
    \theta^* = (X^TX)^{-1}X^Ty
$$

### 2. QR decomposition method

$X$ can be written as $X = QR$ where, (QR decomposition)

$Q$ is an orthogonal matrix of dimension $m x n$ (orthogonal means $Q^TQ = I $) and $R$ is an upper-triangular matrix of dimensions $n * n$. 


$$
    X^T X \theta^* = X^T y
$$
$$
    R^T Q^T Q R \theta^* = R^T Q^T y
$$
$$
  R \theta^* = Q^T y.
$$

The QR decomposition method is more stable and accurate method than the normal equation method.
