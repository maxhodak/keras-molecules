from __future__ import print_function

import argparse
import os
import numpy as np

from pylab import figure, axes, scatter, title, show, savefig

from rdkit import Chem
from rdkit.Chem import Draw

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='Path for data file to read')
    
    return parser.parse_args()

def main():
    args = get_arguments()
    
    data = np.loadtxt(args.data, delimiter='\t')
    
    figure(figsize=(6, 6))
    scatter(data[:, 0], data[:, 1], marker='.')
    savefig('foo.png', bbox_inches='tight')

if __name__ == '__main__':
    main()

