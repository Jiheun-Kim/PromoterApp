# Model card: Week 8 course model

## Model source
notebooks/JiheunKim_mlgenomics_week09_lab.ipynb (renamed from Week 8), evaluated
`clf = grid_search.best_estimator_`; exported directly in the final cell.
RandomForestClassifier(n_estimators=300, max_depth=None, random_state=0, n_jobs=-1).
Remaining settings use the same installed scikit-learn defaults.
No new model selection or hyperparameter tuning is performed.
The separate average-precision search at the end of Week 8 is not the evaluated clf.
Restart & Run All runs the original coursework from the same CSV, split and seed;
the final cell exports the fitted classifier, vectorizer, columns and threshold as
a dictionary in model.joblib. No notebook-defined class/function is serialized.

## Features
Use data/vectorizer.pkl, its original 64 character 3-mer columns,
then GC percentage rounded to 2 decimals (65 total). No new vocabulary or scaler
is fitted. The adapter reproduces every value of the supplied feature matrix.
The notebook and app construct the same 65 input features and reproduce all held-out
probabilities of the original-matrix classifier exactly. Feature construction is in
the final notebook cell; app.py repeats the short calculation for new inputs.
No week8_features.py module is required. Duplicate/N filtering and row order match Week 8.

## Evaluation
Same stratified 80/20 split, random_state=0: 427 training / 107 test sequences.
PR-AUC (average precision): 0.892781822388.
ROC-AUC: 0.900000000000.
Positive prevalence: 55/107 (51.4%). Threshold: 0.30 from Week 8.
Confusion matrix at 0.30: [[32, 20], [1, 54]].
These reproduce the saved Week 8 evaluation outputs.

## Explanation and limitations
Contributions are changes in positive-class probability along each tree's decision
path, averaged over the forest. Baseline + all 65 contributions equals the prediction.
They are path-based attributions, not SHAP or causal motifs. The app shows top
k-mer contributions; GC is used by the classifier but its contribution is not displayed.
The classifier itself is unchanged. Training CSVs are under data/; exported model
and metrics remain in the project root.
Potential related-sequence overlap, small sample size and repeated holdout inspection
limit generalization. Probabilities are uncalibrated. Training sequences are mostly
501 bases; other lengths may be outside the training distribution.
Original dataset provenance/license are not documented. Educational use only.
Only load trusted joblib files. Package versions are recorded in requirements.txt.
