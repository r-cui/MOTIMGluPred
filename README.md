# TimlGluPred
This is the repository for our work glucose excursion prediction using time-index meta-learning.
## Prepare Data
Raw data of the two datasets should be acquired from their raw repositories.

* OhioT1DM: http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html
* UMT1DM: https://github.com/igfox/multi-output-glucose-forecasting/

1. prepare the following structure.
```
storage/
+-- datasets/
|   +-- ohiot1dm/    
|   |   +-- OhioT1DM-training/
|   |   +-- OhioT1DM-testing/
|   |   +-- OhioT1DM-2-training/
|   |   +-- OhioT1DM-2-testing/
|   +-- umt1dm/
|   |   +-- unprocessed_cgm_data.xlsx
```
2. Preprocessing the data using the following commands, which would output the preprocessed data files `{ID}_{train/test}.csv` under `storage/datasets/{dataset}/preprocessed/`.
```bash
python -m storage.datasets.ohiot1dm.preprocess
python -m storage.datasets.umt1dm.preprocess
```

## Requirements

Dependencies for this project can be installed by
```bash
pip install -r requirements.txt
```

## Usage
One can directly run our experiment using the script `train_model.sh`, where the GPU device number and the experiment name (config file name in `experiments/configs`) should be provided as arguments. For example:
```bash
bash train_model.sh 0 ohiot1dm_self
```
The results will be saved to `storage/experiments/`.

## Acknowledgements
Our implementation is based on resources from the following repository, we thank the original
authors for open-sourcing their work.

* https://github.com/salesforce/DeepTime
* https://github.com/igfox/multi-output-glucose-forecasting
