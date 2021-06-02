from random import random
import math
import numpy as np

NO_OF_HIDDEN_NEURONS = 2


class ModelMultilayerPerceptron:

    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x))

    # initialisation of the weights for each neuron of all the layers
    def netInitialisation(self, noInputs, noOutputs, noHiddenNeurons):
        net = []

        hiddenLayer = [Neuron([random() for i in range(noInputs + 1)]) for h in range(noHiddenNeurons)]
        net.append(hiddenLayer)

        outputLayer = [Neuron([random() for i in range(noHiddenNeurons + 1)]) for o in range(noOutputs)]
        net.append(outputLayer)
        return net

    def activate(self, input, weights):
        result = 0.0
        for i in range(0, len(input)):
            result += input[i] * weights[i]
        result += weights[len(input)]
        return result

    # neuron transfer
    def transfer(self, value):
        return self.sigmoid(value)

    # neuron computation/activation
    def forwardPropagation(self,net, inputs):
        for layer in net:
            newInputs = []
            for neuron in layer:
                activation = self.activate(inputs, neuron.weights)
                neuron.output = self.transfer(activation)
                newInputs.append(neuron.output)
            inputs = newInputs
        return inputs

    # inverse transfer of a neuron
    def transferInverse(self, val):
        return val * (1 - val)

    # error propagation
    def backwardPropagation(self, net, expected):
        for i in range(len(net) - 1, 0, -1):
            crtLayer = net[i]
            errors = []
            if i == len(net) - 1:
                # last layer
                for j in range(0, len(crtLayer)):
                    crtNeuron = crtLayer[j]
                    errors.append(expected[j] - crtNeuron.output)
            else:
                # hidden layers
                for j in range(0, len(crtLayer)):
                    crtError = 0.0
                    nextLayer = net[i + 1]
                    for neuron in nextLayer:
                        crtError += neuron.weights[j] * neuron.delta
                    errors.append(crtError)
            for j in range(0, len(crtLayer)):
                crtLayer[j].delta = errors[j] * self.transferInverse(crtLayer[j].output)

    # change the weights
    def updateWeights(self, net, example, learningRate):
        for i in range(0, len(net)):
            inputs = example[:-1]
            if (i > 0):
                # hidden layers or output layer
                # computed values of precedent layer
                inputs = [neuron.output for neuron in net[i - 1]]
            for neuron in net[i]:
                # update weight of all neurons of the current layer
                for j in range(len(inputs)):
                    neuron.weights[j] += learningRate * neuron.delta * inputs[j]
            neuron.weights[-1] += learningRate * neuron.delta

    def trainingMLP(self, net, data, noOutputTypes, learningRate, noEpochs):
        for epoch in range(0, noEpochs):
            for i, example in enumerate(data[0]):
                inputs = example
                computedOutputs = self.forwardPropagation(net, inputs)
                expected = [data[1][i]]
                crtErr = sum([(expected[i] - computedOutputs[i]) ** 2 for i in range(0, len(expected))]) / len(expected)
                # print("Epoch: ", epoch, " example: ", example, " expected: ", expected, " computed: ",computedOutputs, " crtErr: ", crtErr)
                self.backwardPropagation(net, expected)
                self.updateWeights(net, example, learningRate)

    def evaluatingMLP(self,net, data, noOutputTypes):
        computedOutputs = []
        for inputs in data[0]:
            computedOutput = self.forwardPropagation(net, inputs)
            computedOutputs.append(computedOutput[0])
        return computedOutputs

    def computePerformanceRegression(self, computedOutputs, realOutputs):
        error = 0
        for i, pred in enumerate(computedOutputs):
            # print("[%s] Real: %s Predicted: %s" % (i, self.output_data[i], pred))
            error += (pred - realOutputs[i]) ** 2
        return math.sqrt(error / len(computedOutputs))


class Neuron:

    def __init__(self, w = [], out = None, delta = 0.0):
         self.weights = w
         self.output = out
         self.delta = delta

    def __str__(self):
        return "weights: " + str(self.weights) + ", output: " + str(self.output) + ", delta: " + str(self.delta)