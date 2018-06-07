import numpy as np
import cma
from PIL import Image
from const import *


def compute_weight_decay(weight_decay, model_param_list):
  model_param_grid = np.array(model_param_list)
  return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


def _formate_img(img):
    img = np.array(img) * 255
    img = Image.fromarray(np.uint8(img)).resize((WIDTH, HEIGHT))
    return np.array(img).transpose((2, 0, 1))


class CMAES:
    def __init__(self, num_params, sigma_init=0.5, popsize=255, weight_decay=0.0):
        """
        Sigma: initial standard deviation
        Popsize: size of the population
        Num_params: number of parameters in the candidate solution
        Weight_decay: modify the reward according the a weighted mean of the parameters
        """

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.popsize = popsize
        self.weight_decay = weight_decay
        self.solutions = None
        self.es = cma.CMAEvolutionStrategy(self.num_params * [0],
                                           self.sigma_init,
                                           {'popsize': self.popsize })


    def rms_stdev(self):
        sigma = self.es.result[6]
        return np.mean(np.sqrt(sigma*sigma))


    def ask(self):
        """ Ask the CMA-ES to sample new candidate solutions """

        self.solutions = np.array(self.es.ask())
        return self.solutions


    def tell(self, reward_table_result):
        """ Give the reward result to the CMA-ES to adapt mu and sigma """

        reward_table = -np.array(reward_table_result)

        ## Compute reward decay if needed
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay

        self.es.tell(self.solutions, (reward_table).tolist())


    def current_param(self):
        """ Mean solution, better with noise """

        return self.es.result[5]


    def best_param(self):
        """ Best evaluated solution, useful for saving the best agent """

        return self.es.result[0]


    def result(self):
        """ Best params so far, along with historically best reward, curr reward, sigma """

        r = self.es.result
        return (r[0], -r[1], -r[1], r[6])
