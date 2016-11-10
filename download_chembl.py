import import_smiles
import os
import argparse
import urllib

DEFAULT_URI = 'ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_22_chemreps.txt.gz'

def get_arguments():
    parser = argparse.ArgumentParser(description='Download ChEMBL entries and convert them to input for preprocessing')
    parser.add_argument('--uri', type=str, default=DEFAULT_URI,
                        help = 'URI to download ChEMBL entries from')
    parser.add_argument('--outfile', type=str, help='Output file name', default = 'chembl_smiles.h5')
    return parser.parse_args()

def main():
    args = get_arguments()
    fname = os.path.basename(args.uri)

    urllib.urlretrieve(args.uri, fname)

    d = import_smiles.read_smiles(fname, column=1)
    import_smiles.create_h5(d, args.outfile)

if __name__ == '__main__':
    main()
