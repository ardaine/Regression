import numpy
from numpy.linalg import *

trainfile = '/host/Workspace/PythonWorkPlaceLinux/Examples/features-train.txt'
testfile = '/host/Workspace/PythonWorkPlaceLinux/Examples/features-eval.txt'

# Load data from file into numpy array
traindata = numpy.genfromtxt(trainfile)
testdata = numpy.genfromtxt(testfile)

# Add a column of ones as Feature 0, as the bias term
trainInput = numpy.hstack (( numpy.ones((len(traindata), 1)),   traindata[:, 1::]))
testInput = numpy.hstack (( numpy.ones((len(testdata), 1)),   testdata))
trainTarget = traindata[:, 0]

# Train a linear regression model. Use the Normal Equation to find optimal values of the hypothesis parameters,
# since the data is not large
theta =  numpy.dot( numpy.dot(  inv(numpy.dot(trainInput.T, trainInput)), trainInput.T), trainTarget)

# Predicted value
testOutR = numpy.dot(theta.T, testInput.T).T

print testOutR
