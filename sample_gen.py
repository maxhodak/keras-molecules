from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import sys

from molecules.model import MoleculeVAE
from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset
from molecules.vectorizer import SmilesDataGenerator

LATENT_DIM = 292
NUM_SAMPLED = 100
TARGET = 'autoencoder'

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('data', type=str, help='File of latent representation tensors for decoding.')
    parser.add_argument('model', type=str, help='Trained Keras model to use.')
    parser.add_argument('--save_h5', type=str, help='Name of a file to write HDF5 output to.')
    parser.add_argument('--target', type=str, default=TARGET,
                        help='What model to sample from: autoencoder, encoder, decoder.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--sample', type=int, metavar='N', default=NUM_SAMPLED,
                        help='Number of items to sample from data generator.')
    return parser.parse_args()

def read_latent_data(filename):
    h5f = h5py.File(filename, 'r')
    data = h5f['latent_vectors'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    return (data, charset)

def read_smiles_data(filename):
    import pandas as pd
    h5f = pd.read_hdf(filename, 'table')
    data = h5f['structure'][:]
    # import gzip
    # data = [line.split()[0].strip() for line in gzip.open(filename) if line]
    return data

def autoencoder(args, model):
    latent_dim = args.latent_dim

    structures = read_smiles_data(args.data)

    datobj = SmilesDataGenerator(structures, 120)
    train_gen = datobj.generator(1)

    if os.path.isfile(args.model):
        model.load(datobj.chars, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    true_pred_gen = (((mat, weight, model.autoencoder.predict(mat))
                      for (mat, _, weight) in train_gen))
    text_gen = ((str.join('\n',
                          [str((datobj.table.decode(true_mat[vec_ix])[:np.argmin(weight[vec_ix])],
                                datobj.table.decode(vec)[:]))
                           for (vec_ix, vec) in enumerate(pred_mat)]))
                for (true_mat, weight, pred_mat) in true_pred_gen)
    for _ in range(args.sample):
        print(text_gen.next())

def decoder(args, model):
    latent_dim = args.latent_dim
    data, charset = read_latent_data(args.data)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    for ix in range(len(data)):
      sampled = model.decoder.predict(data[ix]).argmax(axis=2)[0]
      sampled = decode_smiles_from_indexes(sampled, charset)
      print(sampled)

def encoder(args, model):
    latent_dim = args.latent_dim

    structures = read_smiles_data(args.data)

    datobj = SmilesDataGenerator(structures, 120)
    train_gen = datobj.generator(1)

    if os.path.isfile(args.model):
        model.load(datobj.chars, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    true_pred_gen = (((mat, weight, model.encoder.predict(mat))
                      for (mat, _, weight) in train_gen))
    if args.save_h5:
        h5f = h5py.File(args.save_h5, 'w')
        h5f.create_dataset('charset', data = datobj.chars)
        h5f.create_dataset('latent_vectors', (args.sample, 120, latent_dim))
        for ix in range(args.sample):
          _, _, x_latent = true_pred_gen.next()
          h5f['latent_vectors'][ix] = x_latent[0]
        h5f.close()
    else:
        text_gen = ((str.join('\n',
                              [str((datobj.table.decode(true_mat[vec_ix])[:np.argmin(weight[vec_ix])],
                                    (vec)[:]))
                              for (vec_ix, vec) in enumerate(pred_mat)]))
                    for (true_mat, weight, pred_mat) in true_pred_gen)
        for _ in range(args.sample):
            print(text_gen.next())

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
