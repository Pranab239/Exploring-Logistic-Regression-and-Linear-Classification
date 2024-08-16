import numpy as np

class LogisticRegression:
    def __init__(self):
        """
        Initialize `self.weights` properly. 
        Recall that for binary classification we only need 1 set of weights (hence `num_classes=1`).
        We have given the default zero intialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 1
        self.d = 2
        self.weights = np.random.randn(self.d+1, self.num_classes)*0.1
        self.v = np.zeros_like(self.weights)
    
    def preprocess(self, input_x):
        """
        Preprocess the input any way you seem fit.
        """
        return input_x

    def sigmoid(self, x):
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        return 1 / (1 + np.exp(-x))

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        N = input_x.shape[0]
        input_x = np.hstack((np.ones((N,1)), input_x))
        output_y = self.sigmoid(np.dot(input_x,self.weights))
        loss = (np.dot(input_y.T, np.log10(output_y)) + np.dot((1 - input_y.T), np.log10(1 - output_y))) * ( - 1 / N)
        return loss[0]

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        N = input_x.shape[0];
        input_x = np.hstack((np.ones((N,1)), input_x))
        output_y = self.sigmoid(np.dot(input_x,self.weights))
        grad = (np.dot((input_y - output_y.T), input_x) * (-1/N))
        grad = grad.T
        return grad

    def update_weights(self, grad, learning_rate, momentum):
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """
        self.v = momentum*self.v - learning_rate*grad
        self.weights += self.v

    def get_prediction(self, input_x):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,) 
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        N = input_x.shape[0]
        input_x = np.hstack((np.ones((N,1)), input_x))
        output_y = self.sigmoid(np.dot(input_x,self.weights))
        file="./submission/best_binary.weights.npy"
        np.save(file, self.weights)
        pred = (output_y > 0.5).astype(int)
        return pred.flatten()

class LinearClassifier:
    def __init__(self):
        """
        Initialize `self.weights` properly. 
        We have given the default zero intialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 3
        self.d = 4
        self.weights = np.random.randn(self.num_classes, self.d+1)*0.01
        self.v = np.zeros_like(self.weights)
    
    def preprocess(self, train_x):
        """
        Preprocess the input any way you seem fit.
        """
        return train_x

    def sigmoid(self, x):
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        pass

    def softmax(self, x):
        """
        Softmax Function for Multinomial Logistic Regression
        """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True) # Normalize the value between 0 and 1.

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        
        N = input_x.shape[0]
        input_x = np.hstack((np.ones((N,1)), input_x))  # Adding one column in the front.
        output_y = self.softmax(np.dot(input_x,self.weights.T)) # Do X.W and put in the softmax function.
        loss = -np.sum(np.log(output_y[range(N), input_y])) / N # L = yi * log(yi')
        return loss

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        N = input_x.shape[0]
        input_x = np.hstack((np.ones((N,1)), input_x))
        output_y = self.softmax(np.dot(input_x, self.weights.T))
        ohe = np.eye(self.num_classes)[input_y] # one hot encoding of the actual label Y.
        gradient = - np.dot(input_x.T, (ohe - output_y)).T / N
        return gradient

    def update_weights(self, grad, learning_rate, momentum):
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """
        self.v = momentum*self.v - learning_rate*grad
        self.weights += self.v

    def get_prediction(self, input_x):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,) 
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        N = input_x.shape[0]
        input_x = np.hstack((np.ones((N,1)), input_x))
        predictions = np.argmax(np.dot(input_x,self.weights.T), axis=1)
        return predictions
