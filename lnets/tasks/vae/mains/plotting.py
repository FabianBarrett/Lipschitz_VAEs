import matplotlib.pyplot as plt
import numpy as np 
import os

def plot_ELBOs():
    lipschitz_constant = np.array([1, 2, 5, 10, 15, 20, 25, 30, 35])
    final_training_ELBO = -np.array([15859325, 11734584, 7866443, 6239877, 5623308, 5362897, 5166768, 5094954, 5050044])
    final_val_ELBO = -np.array([15587363, 11558265, 7800528, 6195488, 5625618, 5365766, 5172807, 5130746, 5103014])
    plt.plot(lipschitz_constant, final_training_ELBO, 'b', label='Training')
    plt.plot(lipschitz_constant, final_val_ELBO, 'r', label='Validation')
    plt.title("ELBO after 10 epochs \n (no finetuning, 3 FC layers)")
    plt.xlabel("Lipschitz constant of encoder mean, encoder variance and decoder")
    plt.ylabel("ELBO")
    plt.legend()
    plt.savefig(os.path.join('./out/vae/binarized_mnist/figures', 'ELBO_Lipschitz_Constant.png'))
    plt.clf()

def plot_deeper_shallower_ELBOs():
    lipschitz_constant = np.array([1, 2, 5, 10, 15, 20, 25, 30, 35])
    shallower_training_ELBO = -np.array([15859325, 11734584, 7866443, 6239877, 5623308, 5362897, 5166768, 5094954, 5050044])
    shallower_val_ELBO = -np.array([15587363, 11558265, 7800528, 6195488, 5625618, 5365766, 5172807, 5130746, 5103014])
    deeper_training_ELBO = -np.array([13718303, 11256677])
    deeper_val_ELBO = -np.array([13490988, 11136993])
    plt.plot(lipschitz_constant, shallower_training_ELBO, 'bo', label='Training, 3 FC layers')
    plt.plot(lipschitz_constant, shallower_val_ELBO, 'ro', label='Validation, 3 FC layers')
    plt.plot(lipschitz_constant[:2], deeper_training_ELBO, 'b+', label='Training, 5 FC layers')
    plt.plot(lipschitz_constant[:2], deeper_val_ELBO, 'r+', label='Training, 5 FC layers')
    plt.title("ELBO after 10 epochs \n (no finetuning)")
    plt.xlabel("Lipschitz constant of encoder mean, encoder variance and decoder")
    plt.ylabel("ELBO")
    plt.legend()
    plt.savefig(os.path.join('./out/vae/binarized_mnist/figures', 'ELBO_Lipschitz_Constant_Deeper_Shallower.png'))
    plt.clf()


if __name__ == "__main__":
    plot_ELBOs()
    plot_deeper_shallower_ELBOs()