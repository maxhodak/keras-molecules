import os
import argparse
import urllib
import pandas
import tempfile
from progressbar import ProgressBar, Percentage, Bar, ETA, FileTransferSpeed

DEFAULTS = {
    "chembl22": {
        "uri": "ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_22/archived/chembl_22_chemreps.txt.gz",
        "outfile": "data/chembl22.h5"
    },
    "zinc12": {
        "uri": "http://zinc.docking.org/db/bysubset/13/13_prop.xls",
        "outfile": "data/zinc12.h5"
    }
}

def get_arguments():
    parser = argparse.ArgumentParser(description = 'Download ChEMBL entries and convert them to input for preprocessing')
    parser.add_argument('--dataset', type = str, help = "%s  ...or specify your own --uri" % ", ".join(DEFAULTS.keys()))
    parser.add_argument('--uri', type = str, help = 'URI to download ChEMBL entries from')
    parser.add_argument('--outfile', type = str, help = 'Output file name')
    args = parser.parse_args()

    if args.dataset and args.dataset in DEFAULTS.keys():
        uri = DEFAULTS[args.dataset]['uri']
        outfile = args.outfile or DEFAULTS[args.dataset]['outfile']
    elif args.dataset not in DEFAULTS.keys():
        parser.error("Dataset %s unknown. Valid choices are: %s" % (args.dataset, ", ".join(DEFAULTS.keys())))
    else:
        uri = args.uri
        outfile = args.outfile
    if uri is None:
        parser.error("You must choose either a known --dataset or provide a --uri and --outfile.")
        sys.exit(1)
    if outfile is None:
        parser.error("You must provide an --outfile if using a custom --uri.")
        sys.exit(1)
    dataset = args.dataset
    return (uri, outfile, dataset)

def main():
    uri, outfile, dataset = get_arguments()
    fd = tempfile.NamedTemporaryFile()
    progress = ProgressBar(widgets=[Percentage(), ' ', Bar(), ' ', ETA(), ' ', FileTransferSpeed()])

    def update(count, blockSize, totalSize):
        if progress.maxval is None:
            progress.maxval = totalSize
            progress.start()
        progress.update(min(count * blockSize, totalSize))

    urllib.urlretrieve(uri, fd.name, reporthook = update)
    if dataset == 'zinc12':
        df = pandas.read_csv(fd.name, delimiter = '\t')
        df = df.rename(columns={'SMILES':'structure'})
        df.to_hdf(outfile, 'table', format = 'table', data_columns = True)
    elif dataset == 'chembl22':
        df = pandas.read_table(fd.name,compression='gzip')
        df = df.rename(columns={'canonical_smiles':'structure'})
        df.to_hdf(outfile, 'table', format = 'table', data_columns = True)
        pass
    else:
        df = pandas.read_csv(fd.name, delimiter = '\t')
        df.to_hdf(outfile, 'table', format = 'table', data_columns = True)

if __name__ == '__main__':
    main()
