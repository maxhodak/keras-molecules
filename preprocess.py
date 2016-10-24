import argparse
import pandas
import h5py
import numpy as np
from autoencoder.utils import one_hot_array, one_hot_index

from sklearn.cross_validation import train_test_split

def get_arguments():
    parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('infile', type=str, help='Input file name')
    parser.add_argument('outfile', type=str, help='Output file name')
    return parser.parse_args()


def prepare_dataset(in_filename, out_filename):
    data = pandas.read_hdf(in_filename, 'table')
    # keys = data['structure'].map(len) < 60
    # assay_keys = data['assay_id'] == 764847 & keys

    structures = data['structure'][0:500000].map(lambda x: list(x.ljust(120)))
    # activities = data['standard_value'][keys]

    del data

    charset = list(reduce(lambda x, y: set(y) | x, structures, set()))

    string_encoded  = np.array(map(lambda row: one_hot_index(row, charset), structures)).reshape(-1, 60)
    one_hot_encoded = np.array(
        map(lambda row:
            map(lambda x: one_hot_array(x, len(charset)),
                one_hot_index(row, charset)),
            structures))

    data_train, data_test = train_test_split(one_hot_encoded, test_size=0.20)

    h5f = h5py.File(out_filename, 'w')
    h5f.create_dataset('charset', data = charset)
    h5f.create_dataset('data_train', data = data_train)
    h5f.create_dataset('data_test', data = data_test)
    h5f.close()

    del one_hot_encoded

def main():
    args = get_arguments()
    prepare_dataset(args.infile, args.outfile)

if __name__ == '__main__':
    main()
