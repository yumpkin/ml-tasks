import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """

    # YOUR CODE HERE
    EPS = 0.0005
    y_mean = np.mean(y, axis=0)
    h = -np.sum(y_mean * np.log2(y_mean + EPS))
    return h

    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE
    p = np.mean(y, axis=0)
    gini = 1 - np.sum(p**2)
    return gini

    # calculate the probability for for each unique label
    # probabilities = []
    # for one_class in np.unique(y):
    #     proba = y[y == one_class].shape[0] / y.shape[0]
    #     probabilities.append(proba)
    # p = np.asarray(probabilities)
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    # YOUR CODE HERE
    R = len(y)
    var = 1/R * sum((y - np.mean(y))**2)
    return var

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    R = len(y)
    mad_med = 1/R * sum(np.abs(y - np.median(y)))
    return mad_med


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """
        # YOUR CODE HERE
        value_of_feat = X_subset[:, feature_index]
        X_left, X_right = X_subset[value_of_feat < threshold, :], X_subset[value_of_feat >= threshold, :]
        y_left, y_right = y_subset[value_of_feat < threshold], y_subset[value_of_feat >= threshold]
        
        # X_left, X_right = list(), list()
        # for value_of_feat in X_subset:
        #   if value_of_feat[:, feature_index] < threshold:
        #     X_left.append(value_of_feat)
        #   else:
        #     X_right.append(value_of_feat)

        # y_left, y_right = list(), list()
        # for class_label in y_subset:
        #   if class_label[:, feature_index] < threshold:
        #     y_left.append(class_label)
        #   else:
        #     y_right.append(class_label)
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        value = X_subset[:, feature_index]
        y_left, y_right = y_subset[value <= threshold], y_subset[value > threshold]
        
        # y_left, y_right = list(), list()
        # for class_label in y_subset:
        #   if class_label[:, feature_index] < threshold:
        #     y_left.append(y)
        #   else:
        #     y_right.append(y)
            
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE
        feature_index = None
        best_threshold = None
        optimal_split_fuctional = float('-inf')
        number_of_features = X_subset.shape[1]
        # for each column in X
        for feat_index in range(number_of_features):
            feat = np.unique(X_subset[:, feat_index])
            # for each value in the feature column
            for value in feat:
                y_left, y_right = self.make_split_only_y(feat_index, value, X_subset, y_subset)
                # discard those that have not make split
                if y_right.shape[1] == 0 or y_left.shape[1] == 0:
                    continue
                    
                # calculate impurity for the right and left nodes
                impurityRight = self.criterion(y_right)
                impurityLeft = self.criterion(y_left)

                #just for more meaningful name while we iterate through nodes
                dataset_current_node = number_of_features

                left_subset = y_left.shape[1]
                right_subset = y_right.shape[1]
                
                split_fuctional = (impurityLeft * left_subset / dataset_current_node)+(impurityRight * right_subset / dataset_current_node)
                
                # is this split_fuctional is the optimal one so fat?
                if split_fuctional > optimal_split_fuctional:
                    feature_index = feat_index
                    best_threshold = value
                    optimal_split_fuctional = split_fuctional

        return feature_index, best_threshold
    
#         if self.criterion_name = 'gini':
#             impurity = gini(y_subset)
#             for feature_index in X_subset:
#                 for value in y_subset[feature_index]:
#                     threshold = value
#                     impurity_new = gini(y_subset[feature_index])
#                     if impurity_new < impurity:

    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        depth = 0
        n_features = X_subset.shape[0]
        if (depth < self.max_depth) and (n_features >= self.min_samples_split):
          # Getting the best split
          feature_index, threshold = self.choose_best_split(X_subset, y_subset)
          (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)

          new_node = Node(feature_index, threshold)

          new_node.right_child = self.make_tree(X_right, y_right)
          self.max_depth += 1
          new_node.left_child = self.make_tree(X_left, y_left)
          self.max_depth += 1

        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
    
    @staticmethod
    def current_node_info(root, X):
        current_node = root
        for feat_value in X:
          while current_node.proba is None:
            given_feature_value = feat_value[current_node.feature_index]
            if given_feature_value > current_node.value:
              current_node = current_node.right_child
            else:
              current_node = current_node.left_child
        return current_node


    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        # YOUR CODE HERE
        n_objects = X.shape[0]
        y_predicted = np.zeros((n_objects, 1))
        current_node = DecisionTree.current_node_info(self.root, X)
        if self.classification:
          np.append(y_predicted, np.argmax(current_node.proba))
        else:
          np.append(y_predicted, current_node.proba)

        return np.asarray(y_predicted)
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE
        n_objects = X.shape[0]
        y_predicted_probs = np.zeros((n_objects, self.n_classes))
        current_node = DecisionTree.current_node_info(X)
        np.append(y_predicted_probs, current_node.proba)
        
        return y_predicted_probs
