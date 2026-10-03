# Data provenance

## ExtraSensory primary release

- Official project: <https://extrasensory.ucsd.edu/>
- Archive: `ExtraSensory.per_uuid_features_labels.zip`
- Official URL:
  <http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip>
- SHA-256:
  `412D307BDF54A60DFF81A705C2F25833889A46DCA64A184E3B588AD99C69CB56`
- Retrieved: 2026-09-21

The archive contains 60 participant-specific compressed CSV files. The local
profile found 377,346 rows, 225 sensor/context features, and 51 cleaned labels.

## Official cross-validation partition

- Official URL: <http://extrasensory.ucsd.edu/data/cv5Folds.zip>
- Local extraction: `data/raw/cv5Folds/cv_5_folds/`

The files split all 60 participants into five test folds, stratified by phone
platform. The experiment uses one fold for outer testing and a different fold
for validation, leaving 36/12/12 train/validation/test users.

## Citation requirement

Any publication using the dataset must cite:

Y. Vaizman, K. Ellis, and G. Lanckriet, “Recognizing Detailed Human Context
In-the-Wild from Smartphones and Smartwatches,” *IEEE Pervasive Computing*,
vol. 16, no. 4, pp. 62–74, 2017.

The official release and all generated artifacts under `data/raw/`,
`data/processed/`, and `results/` are excluded from Git.

