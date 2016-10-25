from __future__ import print_function

import argparse
import os
import numpy as np

from autoencoder.model import MoleculeVAE
from autoencoder.utils import load_dataset

from pylab import figure, axes, scatter, title, show, savefig

from rdkit import Chem
from rdkit.Chem import Draw

from keras.utils.visualize_util import plot

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--data', type=str, help='Path for data file to read')
    parser.add_argument('--model', type=str, help='Path for model file to visualize')

    return parser.parse_args()

def visualize_model(args):
    model = MoleculeVAE()

    data, charset = load_dataset(args.data, split = False)

    if os.path.isfile(args.model):
        model.load(charset, args.model)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    plot(model.autoencoder, to_file='model.png')

def plot_2d(args):
    data = np.loadtxt(args.data, delimiter='\t')
    figure(figsize=(6, 6))
    scatter(data[:, 0], data[:, 1], marker='.', linewidth='0', s = 0.2)
    savefig('foo.png', bbox_inches='tight')

def main():
    args = get_arguments()

    if args.model:
        visualize_model(args)

    elif args.data:
        plot_2d(args)

if __name__ == '__main__':
    main()
