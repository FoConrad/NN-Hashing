import torch
from .mnist_base import MNISTAttack
import math
from torch.autograd.gradcheck import zero_gradients

# Code taken from https://github.com/ast0414/adversarial-example/blob/master/craft.py

class MNIST_JSMA(MNISTAttack):
    def __init__(self, model_class, weights_file):
        super().__init__(model_class, weights_file,fgsm=True)

    def compute_jacobian(self,inputs, output):
        """
        :param inputs: Batch X Size (e.g. Depth X Width X Height)
        :param output: Batch X Classes
        :return: jacobian: Batch X Classes X Size
        """
        assert inputs.requires_grad

        num_classes = output.size()[1]

        jacobian = torch.zeros(num_classes, *inputs.size())
        grad_output = torch.zeros(*output.size())
        if inputs.is_cuda:
            grad_output = grad_output.cuda()
            jacobian = jacobian.cuda()

        for i in range(num_classes):
            zero_gradients(inputs)
            grad_output.zero_()
            grad_output[:, i] = 1
            output.backward(grad_output, retain_variables=True)
            jacobian[i] = inputs.grad.data

        return torch.transpose(jacobian, dim0=0, dim1=1)

    def saliency_map(self,jacobian, search_space, target_index, increasing=True):
        all_sum = torch.sum(jacobian, 0).squeeze()
        alpha = jacobian[target_index].squeeze()
        beta = all_sum - alpha

        if increasing:
            mask1 = torch.ge(alpha, 0.0)
            mask2 = torch.le(beta, 0.0)
        else:
            mask1 = torch.le(alpha, 0.0)
            mask2 = torch.ge(beta, 0.0)

        mask = torch.mul(torch.mul(mask1, mask2), search_space)

        if increasing:
            saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
        else:
            saliency_map = torch.mul(torch.mul(torch.abs(alpha), beta), mask.float())

        max_value, max_idx = torch.max(saliency_map, dim=0)

        return max_value, max_idx


    # TODO: Currently, assuming one sample at each time
    def attack(self,input_tensor,y_true, target_class, max_distortion=0.1):

        # Make a clone since we will alter the values
        input_features = torch.autograd.Variable(torch.FloatTensor(input_tensor).clone(), requires_grad=True)
        input_features = input_features.view(1, 28,28)
        num_features = input_features.size(2)
        max_iter = math.floor(num_features * max_distortion)
        count = 0

        # a mask whose values are one for feature dimensions in search space
        search_space = torch.ones(num_features).byte()
        if input_features.is_cuda:
            search_space = search_space.cuda()

        output = self._model(input_features)
        _, source_class = torch.max(output.data, 1)

        y_pred = source_class[0]

        print('input size:-------')
        print(list(input_features.view(1, 28,28).size()))
        print('ypred:-------------')
        print(y_pred)
        print('true:-----------')
        print(y_true)
        print('source:----------')
        print(source_class[0])
        print('target:-------')
        print(target_class)
        print('max_iter:-----------')
        print(max_iter)
        print('space:---------')
        print(search_space.sum())
        print("************")
        

        

        while (count < max_iter) and (source_class[0] != target_class) and (search_space.sum() != 0):

            print("insideloop")
            # Calculate Jacobian
            jacobian = self.compute_jacobian(input_features, output)

            increasing_saliency_value, increasing_feature_index = self.saliency_map(jacobian, search_space, target_class, increasing=True)

            mask_zero = torch.gt(input_features.data.squeeze(), 0.0)
            search_space_decreasing = torch.mul(mask_zero, search_space)
            decreasing_saliency_value, decreasing_feature_index = self.saliency_map(jacobian, search_space_decreasing, target_class, increasing=False)

            if increasing_saliency_value[0] == 0.0 and decreasing_saliency_value[0] == 0.0:
                break

            if increasing_saliency_value[0] > decreasing_saliency_value[0]:
                input_features.data[0][increasing_feature_index] += 1
            else:
                input_features.data[0][decreasing_feature_index] -= 1

            output = self._model(input_features)
            _, source_class = torch.max(output.data, 1)

            count += 1

        y_pred_adversarial = source_class

        return (input_features.data - torch.FloatTensor(input_tensor)).numpy(),y_pred,y_pred_adversarial
