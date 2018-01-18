from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import sys
import json

from molecules.model import MoleculeVAE
from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset

from pylab import figure, axes, scatter, title, show

from rdkit import Chem
from rdkit.Chem import Draw

LATENT_DIM = 292
TARGET = 'autoencoder'

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='File of latent representation tensors for decoding.')
    parser.add_argument('model', type=str, help='Trained Keras model to use.')
    parser.add_argument('--save_h5', type=str, help='Name of a file to write HDF5 output to.')
    parser.add_argument('--save_json', type=str, help='Name of a file to write json output to.')
    parser.add_argument('--target', type=str, default=TARGET,
                        help='What model to sample from: autoencoder, encoder, decoder.')
    parser.add_argument('--smiles', type=str, default=None,
                        help='A text file of smiles strings.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    return parser.parse_args()

def encoded_to_string(encoded, charset):
    return decode_smiles_from_indexes(map(from_one_hot_array, encoded), charset)

def one_hot_encoded(string, charset):
    return map(lambda x: one_hot_array(x, len(charset)), one_hot_index(string, charset))

def string_to_encoded(string, charset):
    padded = string.ljust(120)
    encoded = one_hot_encoded(padded, list(charset))
    return np.array(encoded)

def read_latent_data(filename):
    h5f = h5py.File(filename, 'r')
    data = h5f['latent_vectors'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    return (data, charset)

def autoencoder(args, model):
    latent_dim = args.latent_dim
    data, charset = load_dataset(args.data, split = False)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    sampled = model.autoencoder.predict(data[0].reshape(1, 120, len(charset))).argmax(axis=2)[0]
    mol = decode_smiles_from_indexes(map(from_one_hot_array, data[0]), charset)
    sampled = decode_smiles_from_indexes(sampled, charset)
    print(mol)
    print(sampled)

def decoder(args, model):
    latent_dim = args.latent_dim
    data, charset = read_latent_data(args.data)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    sampled = model.decoder.predict(data[0].reshape(1, latent_dim)).argmax(axis=2)[0]
    sampled = decode_smiles_from_indexes(sampled, charset)
    print(sampled)

def encoder(args, model):
    latent_dim = args.latent_dim
    data, charset = load_dataset(args.data, split = False)

    if args.smiles is not None:
        # replace data with data from file
        with open(args.smiles) as f:
            lines = f.readlines()
        data = map(lambda x:string_to_encoded(x.rstrip(), charset), lines)
        data = np.array(data)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    x_latent = model.encoder.predict(data)
    if args.save_h5:
        h5f = h5py.File(args.save_h5, 'w')
        h5f.create_dataset('charset', data = charset)
        h5f.create_dataset('latent_vectors', data = x_latent)
        h5f.close()
    elif args.save_json:
        with open(args.save_json, 'w') as outfile:
            json.dump(x_latent.tolist(), outfile)
    else:
        np.savetxt(sys.stdout, x_latent, delimiter = '\t')

def main():
    args = get_arguments()
    model = MoleculeVAE()

    if args.target == 'autoencoder':
        autoencoder(args, model)
    elif args.target == 'encoder':
        encoder(args, model)
    elif args.target == 'decoder':
        decoder(args, model)

if __name__ == '__main__':
    main()
