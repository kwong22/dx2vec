# dx2vec

Modified word2vec skip-gram model to learn embeddings for ICD9 Diagnosis Codes.
Code is adapted from the CS20 class (TensorFlow for Deep Learning Research).

Patient information (subject IDs, ICD9 codes) was obtained from the [MIMIC III Database](https://mimic.physionet.org/).
The list of all ICD9 codes and their descriptions was found at [Centers for Medicare & Medicaid Services](https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/codes.html).

## Required modules
* numpy
* tensorflow
* pandas

## Required data
* Patient data, which should include
  * ICD9 codes
  * Identifiers for the patients corresponding to the ICD9 codes
* All ICD9 Diagnosis Codes, along with text descriptions

Change paths to data files as necessary in the `utils.batch_gen` function.

## Running the model

Build the model and train it:

    $ python dx2vec.py

Run this command and visit [http://localhost:6006](http://localhost:6006) to view the graph structure and training statistics:

    $ tensorboard --logdir='graphs/dx2vec'

Run this command and visit [http://localhost:6006](http://localhost:6006) to visualize the learned representations of the ICD9 codes:

    $ tensorboard --logdir='visualization'

