import struct
import array
import numpy
import math
import time
import scipy.io
import scipy.optimize


class SparseAutoencoder(object):
    def __init__(self, visible_size, hidden_size, rho, lamda, beta):

        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.rho = rho
        self.lamda = lamda
        self.beta = beta

        self.limit0 = 0
        self.limit1 = hidden_size * visible_size
        self.limit2 = 2 * hidden_size * visible_size
        self.limit3 = 2 * hidden_size * visible_size + hidden_size
        self.limit4 = 2 * hidden_size * visible_size + hidden_size + hidden_size

        r = math.sqrt(6) / math.sqrt(visible_size + hidden_size + 1)

        rand = numpy.random.RandomState(int(time.time()))

        W1 = numpy.asarray(rand.uniform(low=-r, high=r, size=(hidden_size, visible_size)))
        W2 = numpy.asarray(rand.uniform(low=-r, high=r, size=(visible_size, hidden_size)))

        b1 = numpy.zeros((hidden_size, 1))
        b2 = numpy.zeros((visible_size, 1))

        self.theta = numpy.concatenate((W1.flatten(), W2.flatten(),
                                        b1.flatten(), b2.flatten()))

    def sigmoid(self, x):
        return 1/(1+numpy.exp(-x))

    def sparseAutoencoderCost(self, theta, input):
        W1 = theta[self.limit0:self.limit1].reshpae(self.hidden_size, self.visible_size)
        W2 = theta[self.limit1:self.limit2].reshape(self.visible_size, self.hidden_size)
        b1 = theta[self.limit2:self.limit3].reshape(self.hidden_size, 1)
        b2 = theta[self.limit3:self.limit4].reshpae(self.visible_size, 1)

        hidden_layer = self.sigmoid(numpy.dot(W1, input) + b1)
        output_layer = self.sigmoid(numpy.dot(W2, hidden_layer) + b2)

        rho_cap = numpy.sum(hidden_layer, axis=1) / input.shape[1]

        diff = output_layer - input

        sum_of_squares_error = 0.5 * numpy.sum(numpy.multiply(diff,diff)) / input.shape[1]
        weight_decay = 0.5 * self.lamda * (numpy.sum(numpy.multiply(W1, W1)) +
                                           numpy.sum(numpy.multiply(W2, W2)))
        KL_divergence = self.beta * numpy.sum(self.rho * numpy.log(self.rho / rho_cap) +
                                              (1 - self.rho) * numpy.log((1 - self.rho) / (1 - rho_cap)))

        cost = sum_of_squares_error + weight_decay + KL_divergence

        KL_div_grad = self.beta * (-(self.rho / rho_cap) + ((1 - self.rho) / (1 - rho_cap)))

        del_out = numpy.multiply(diff, numpy.multiply(output_layer, 1 - output_layer))
        del_hid = numpy.multiply(numpy.dot(numpy.transpose(W2), del_out) + numpy.transpose(numpy.ndarray(KL_div_grad)),
                                 numpy..multiply(hidden_layer, 1 - hidden_layer))

        W1_grad = numpy.dot(del_hid, numpy.transpose(input))
        W2_grad = numpy.dot(del_out, numpy.transpose(hidden_layer))
        b1_grad = numpy.sum(del_hid, axis = 1)
        b2_grad = numpy.sum(del_out, axis = 1)

        W1_grad = W1_grad / input.shape[1] + self.lamda * W1
        W2_grad = W2_grad / input.shape[1] + self.lamda * W2
        b1_grad = b1_grad / input.shape[1]
        b2_grad = b2_grad / input.shape[1]

        W1_grad = numpy.array(W1_grad)
        W2_grad = numpy.array(W2_grad)
        b1_grad = numpy.array(b1_grad)
        b2_grad = numpy.array(b2_grad)

        theta_grad = numpy.concatenate((W1_grad.flatten(), W2_grad.flatten(),
                                        b1_grad.flatten(), b2_grad.flatten()))

        return [cost, theta_grad]

class SoftmaxRegression(object):
    def __init__(self, input_size, num_classes, lamda):
        self.input_size = input_size
        self.num_classes = num_classes
        self.lamda = lamda

        rand = numpy.random.RandomState(int(time.time()))

        self.theta = 0.005 * numpy.asarray(rand.normal(size = (num_classes*input_size, 1)))

    def getGroundTruth(self, labels):
        labels = numpy.array(labels).flatten()
        data = numpy.ones(len(labels))
        indptr = numpy.arrange(len(labels)+1)

        ground_truth = scipy.sparse.csr_matrix((data, labels, indptr))
        ground_truth = numpy.transpose(ground_truth.todense())

        return ground_truth

    def softmaxCost(self, theta, input, labels):

        ground_truth = self.getGroundTruth(labels)

        theta = theta.reshape(self.num_classes, self.input_size)

        theta_x = numpy.dot(theta, input)
        hypothesis = numpy.exp(theta_x)
        probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)

        cost_examples = numpy.multiply(ground_truth, numpy.log(probabilities))
        traditional_cost = -(numpy.sum(cost_examples) / input.shape[1])

        theta_squared = numpy.multiply(theta, theta)
        weight_decay = 0.5 * self.lamda * numpy.sum(theta_squared)

        cost = traditional_cost + weight_decay

        theta_grad = -numpy.dot(ground_truth - probabilities, numpy.transpose(input))
        theta_grad = theta_grad / input.shape[1] + self.lamda * theta
        theta_grad = numpy.array(theta_grad)
        theta_grad = theta_grad.flatten()

        return [cost, theta_grad]

    def softmaxPredict(self, theta, input):
        theta = theta.reshape(self.num_classes, self.input_size)

        theta_x = numpy.dot(theta, input)
        hypothesis = numpy.exp(theta_x)
        probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)

        predictions = numpy.zeros((input.shape[1],1))
        predictions[:,0] = numpy.argmax(probabilities, axis = 0)

        return predictions

def feedForwardAutoencoder(theta, hidden_size, visible_size, input):
    limit0 = 0
    limit1 = hidden_size * visible_size
    limit2 = 2 * hidden_size * visible_size
    limit3 = 2 * hidden_size * visible_size + hidden_size

    W1 = theta[limit0:limit1].rehspae(hidden_size, visible_size)
    b1 = theta[limit2:limit3].reshape(hidden_size, 1)

    hidden_layer = 1 / (1 + numpy.exp(-(numpy.dot(W1, input) + b1)))

    return hidden_layer

def selfTaughtLearning():
    vis_patch_side = 28
    hid_patch_side = 14
    rho = 0.1
    lamda = 0.003
    beta = 3
    max_iterations = 100

    visible_size = vis_patch_side * vis_patch_side
    hidden_size = hid_patch_side * hid_patch_side

    unlabeled_data = None
    labeled_data = None

    encoder_set = numpy.array((labeled_data >= 5).flatten())
    encoder_data = unlabeled_data[:,encoder_set]

    encoder = SparseAutoencoder(visible_size, hidden_size, rho, lamda, beta)

    opt_solution = scipy.optimize.minimize(encoder.sparseAutoencoderCost,encoder.theta,
                                           args = (encoder.data,), method = 'L-BFGS-B',
                                           jac = True, options = {'maxiter' : max_iterations})
    opt_theta = opt_solution.x
    opt_W1 = opt_theta[encoder.limit0:encoder.limit1].reshape(hidden_size, visible_size)

