"""
Author: Sam Armstrong
Date: Autumn 2021

Description: The code for generating a single sample using the model (saves the image to the local folder)
"""

import torch
import numpy as np
from Generator import Generator
from PIL import Image
from matplotlib import cm
from torch.autograd import Variable
from torch import Tensor

device = torch.device('cuda')

def run_model():
    generator = Generator()
    generator.load_state_dict(torch.load('generator-model.pickle'))
    generator.eval()
    
    z = Variable(Tensor(np.random.rand(1, 16)))
    image_array = generator(z).detach().numpy()
    image_array = image_array.reshape(28, 28)
    data = Image.fromarray(image_array)
    data = Image.fromarray(np.uint8(cm.gist_earth(image_array) * 255))
    data.show()
    data.save('GAN-Image.png')

if __name__ == '__main__':
    run_model()
