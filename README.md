# Week 9: Promoter predictor

Uses the evaluated `clf = grid_search.best_estimator_` in
`JiheunKim_mlgenomics_week08_lab.ipynb`: RandomForestClassifier with
n_estimators=300, max_depth=None, random_state=0, n_jobs=-1.
The original Week 8 notebook is now `notebooks/JiheunKim_mlgenomics_week09_lab.ipynb`.
Its original training/evaluation cells are preserved, with the export code appended
at the end. The final cell saves the evaluated `clf` directly; it does not retrain
or replace that classifier. The original stratified 80/20 split uses random_state=0.

## Run

Open `notebooks/JiheunKim_mlgenomics_week09_lab.ipynb` in VS Code and run all with the existing
`genomics-ml` kernel. Restart & Run All executes the original coursework and the
five final export cells (imports, feature construction, checks, model export,
evaluation information), which write model.joblib and metrics.json.
In a terminal with genomics-ml active:

```sh
streamlit run app.py
```

No script installs packages. Versions are in requirements.txt and environment.yml.

## Folder layout

```text
week09/
  app.py
  model.joblib
  metrics.json
  model_card.md
  requirements.txt
  environment.yml
  README.md
  lab9_working_app.jpg
  notebooks/
    JiheunKim_mlgenomics_week09_lab.ipynb
  data/
    skeletal_dataset.csv
    feature_matrix_X.csv
    feature_matrix_y.csv
    vectorizer.pkl
```

The notebook reads from data/ and exports model.joblib and metrics.json to the
project root. It resolves paths when run from either the root or notebooks/.
app.py loads its model and metrics relative to its own location, so it needs no
path change. model_card.md documents the model, evaluation, threshold and limitations;
it is documentation rather than executable code.

## Input and output

Paste one DNA sequence; GC content is computed automatically, not entered separately.
The adapter preserves the supplied vectorizer's 64 3-mer count columns and the 65th
GC-percentage column. Every value is checked against the original feature matrix.
The final notebook cell contains the feature construction and exports a dictionary
with the unchanged classifier, fitted vectorizer, column order and threshold.
The app performs the same 3-mer + GC calculation and reproduces all held-out predictions.
No week8_features.py or separate training script is needed. Prediction explanations
are calculated inside app.py because they must run for each new user input.
PR-AUC: 0.8927818223882641; ROC-AUC: 0.90; positive ratio: 55/107; threshold: 0.30.

The app cleans whitespace/lowercase, accepts one pasted FASTA record or a multi-record
FASTA upload, rejects non-ATGC input, shows probabilities and evaluation metrics,
and provides a threshold slider and downloadable prediction CSV.
DNA cleaning and FASTA parsing are defined directly in app.py; no sequence_utils.py
or other project Python module is required to run the app.
Top k-mers use signed probability changes along forest decision paths. A zero count
can contribute because absence also determines which tree branch is followed.
These are path-based explanations, not SHAP, coefficients or causal motifs.
See model_card.md for limitations. Educational use only.

Original input files are retained. Verification-only scripts/reports, temporary CSVs,
example FASTA and server logs are removed. The working-app screenshot is a Lab 9
submission deliverable. Public deployment is excluded at the user's request.
