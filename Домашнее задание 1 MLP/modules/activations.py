import numpy as np
from .base import Module
from copy import deepcopy
import scipy


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(0, input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        input_copy = input.copy()
        input_copy[input_copy <= 0] = 0
        input_copy[input_copy > 0] = 1
        return grad_output * input_copy

class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return self.sigmoid(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        res = self.sigmoid(input)
        return grad_output * res * (1 - res)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return scipy.special.softmax(input, axis=1)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        m, n = input.shape

        p = scipy.special.softmax(input, axis=1)
        tensor1 = np.einsum('ij,ik->ijk', p, p)
        tensor2 = np.einsum('ij,jk->ijk', p, np.eye(n, n))
        dsoftmax = tensor2 - tensor1
        dz = np.einsum('ijk,ik->ij', dsoftmax, grad_output)
        return dz


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return scipy.special.log_softmax(input, axis=1)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        softmax = Softmax()
        dl_dx = grad_output / softmax.compute_output(input)
        dl_dinput = softmax.compute_grad_input(input, dl_dx)
        return dl_dinput

