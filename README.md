# A Keras implementation of Aspuru-Guzik's molecular autoencoder paper

<table style="border-collapse: collapse">
<tr>
<td style="vertical-align: top" valign="top">
    <strong>Abstract from the paper</strong>
    <p>We report a method to convert discrete representations of molecules to and from a multidimensional continuous representation. This generative model allows efficient search and optimization through open-ended spaces of chemical compounds.</p>
    <p>We train deep neural networks on hundreds of thousands of existing chemical structures to construct two coupled functions: an encoder and a decoder. The encoder converts the discrete representation of a molecule into a real-valued continuous vector, and the decoder converts these continuous vectors back to the discrete representation from this latent space.</p>
    <p>Continuous representations allow us to automatically generate novel chemical structures by performing simple operations in the latent space, such as decoding random vectors, perturbing known chemical structures, or interpolating between molecules. Continuous representations also allow the use of powerful gradient-based optimization to efficiently guide the search for optimized functional compounds. We demonstrate our method in the design of drug-like molecules as well as organic light-emitting diodes.</p>
    <p>
        <strong>Link to the paper</strong><br />
        <a href="https://arxiv.org/abs/1610.02415">arXiv</a>
    </p>
</td><td width="300">
<img src="images/network.png" width="300" /></img>
</td>
</tr>
</table>

## Requirements

Install using `pip install -r requirements.txt`

## Preparing the data

To train the network you need a lot of SMILES strings. The `preprocess.py` script assumes you have an HDF5 file that contains a table structure, one column of which is named `structure` and contains one SMILES string no longer than 120 characters per row. The script then:

- Normalizes the length of each string to 120 by appending whitespace as needed.
- Builds a list of the unique characters used in the dataset. (The "charset")
- Substitutes each character in each SMILES string with the integer ID of its location in the charset.
- Converts each character position to a one-hot vector of len(charset).
- Saves this matrix to the specified output file.

Example:

`python preprocess.py data/dataset.h5 data/processed.h5`

## Training the network

The preprocessed data can be fed into the `train.py` script:

`python train.py data/processed.h5 model.h5 --epochs 20`

If a model file already exists it will be opened and resumed. If it doesn't exist, it will be created.

## Sampling from a trained model

There are two scripts here for sampling from a trained model.

- `sample.py` is useful for just testing the autoencoder.
- `sample_latent.py` will yield the value of the `Dense(292)` tensor that is the informational bottleneck in the model for visualization or analysis.

Note that when using `sample_latent.py`, the `--visualize` flag will use PCA and t-SNE to fit a manifold using the implementations of those algorithms found in scikit-learn, which tend to fall over on even medium sized datasets. It's recommended to simply get the latent representation from that script and then use something else to visualize it.

Example (using [bh_tsne](https://github.com/lvdmaaten/bhtsne)):

```
python sample_latent.py data/processed.h5 model.h5 > data/latent.dat

cat data/latent.dat | python bhtsne.py -d 2 -p 0.1 > data/result.dat

python plot.py data/result.dat
```

## Performance

After 30 epochs on a 500,000 molecule extract from ChEMBL 21 (~7 hours on a NVIDIA GTX 1080), I'm seeing a loss of 0.26 and a reconstruction accuracy of 0.98.

Projecting the dataset onto 2D latent space gives a figure that looks pretty reasonably like Figure 3 from the paper, though there are some strange striations and it's not quite as well spread out as the examples in the paper.

<img src="images/latent_2d.png" />
