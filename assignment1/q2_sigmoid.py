#!/usr/bin/env python

import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    s = 1/(1+np.exp(-x))
    ### END YOUR CODE

    return s


def sigmoid_grad(s):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input s should be the sigmoid
    function value of your original input x.

    Arguments:
    s -- A scalar or numpy array.

    Return:
    ds -- Your computed gradient.
    """

    ### YOUR CODE HERE
    ds = s*(1-s)
    ### END YOUR CODE

    return ds


def test_sigmoid_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print ("Running basic tests...")
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print (f)
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print (g)
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])

    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print ("You should verify these results by hand!\n")


def test_sigmoid():
    """
    Use this space to test your sigmoid implementation by running:
        python q2_sigmoid.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print ("Running your tests...")
    x = np.array([[1, 2], [-1, -2], [1, 5], [-1, -8]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print (f)
    f_ans = np.array([
        [7.31058579e-01, 8.80797078e-01],
        [2.68941421e-01, 1.19202922e-01],
        [7.31058579e-01, 9.93307149e-01],
        [2.68941421e-01, 3.35350130e-04]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print (g)
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359],
        [0.19661193, 0.00664806],
        [0.19661193, 0.00033524]])

    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)


if __name__ == "__main__":
    test_sigmoid_basic();
    test_sigmoid()
