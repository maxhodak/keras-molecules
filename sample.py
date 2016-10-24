from __future__ import print_function

import argparse
import os
import h5py
import numpy as np

from autoencoder.model import MoleculeVAE
from autoencoder.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset

from pylab import figure, axes, scatter, title, show

from rdkit import Chem
from rdkit.Chem import Draw

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('model', type=str, help='Trained Keras model to use.')
    
    return parser.parse_args()

def main():
    args = get_arguments()
    model = MoleculeVAE()
    
    data, charset = load_dataset('data/all_smiles_120_one_hot.h5', split = False)
    
    if os.path.isfile(args.model):
        model.load(charset, args.model)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)
    
    sampled = model.autoencoder.predict(data[100].reshape(1, 120, len(charset))).argmax(axis=2)[0]
    
    mol = decode_smiles_from_indexes(map(from_one_hot_array, data[100]), charset)
    sampled = decode_smiles_from_indexes(sampled, charset)
    print(mol)
    print(sampled)
    
    #p = Chem.MolFromSmiles(mol)
    #Draw.MolToMPL(p)

if __name__ == '__main__':
    main()

