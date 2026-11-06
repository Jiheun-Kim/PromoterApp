from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# DNA cleaning and FASTA parsing
def clean_sequence(text):
    lines = text.strip().splitlines()
    if any(line.lstrip().startswith('>') for line in lines):
        records = parse_fasta(text)
        if len(records) != 1:
            raise ValueError('Paste one FASTA record, or upload a file for batch prediction.')
        return records[0][1]
    seq = ''.join(text.split()).upper()
    if not seq:
        raise ValueError('Enter a DNA sequence.')
    invalid = sorted(set(seq) - set('ATGC'))
    if invalid:
        raise ValueError('Only ATGC bases are accepted. Invalid: ' + ', '.join(invalid))
    if len(seq) < 3:
        raise ValueError('At least 3 bases are needed for 3-mers.')
    return seq


def parse_fasta(text):
    records, name, parts = [], None, []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if name is not None:
                records.append((name, clean_sequence(''.join(parts))))
            name, parts = line[1:].strip(), []
            if not name:
                raise ValueError('Each FASTA record needs a non-empty header.')
        else:
            if name is None:
                raise ValueError('FASTA must start with a >header.')
            parts.append(line)
    if name is None:
        raise ValueError('No FASTA records found.')
    records.append((name, clean_sequence(''.join(parts))))
    return records


ROOT = Path(__file__).resolve().parent

@st.cache_resource
def load_model_bundle():
    return joblib.load(ROOT / 'model.joblib')

model = load_model_bundle()
clf, vec = model['clf'], model['vec']
metrics = json.loads((ROOT / 'metrics.json').read_text(encoding='utf8'))
st.title('Promoter predictor')
thr = st.sidebar.slider('Threshold', 0.0, 1.0, float(metrics['threshold']), 0.01)
st.sidebar.caption(f"Held-out PR-AUC (average precision): {metrics['pr_auc']:.3f}")
st.sidebar.caption(
    f"Positive class ratio: {metrics['test_positive']}/{metrics['test_n']} "
    f"({metrics['positive_prevalence']:.1%})")
st.sidebar.caption(f"Current threshold: {thr:.2f}")
st.sidebar.caption(
    f"Random forest: 300 trees, max_depth=None, random_state=0. "
    f"Default threshold: {metrics['threshold']:.2f}")
st.sidebar.caption('Trained on 501 bases so other lengths may be unreliable')
seq = st.text_area('Paste a DNA sequence', help='ATGC only; lowercase, whitespace and one FASTA header are cleaned.')
# Hide Streamlit's automatic upload-size/type caption while keeping the upload control.
st.markdown(
    '<style>[data-testid="stFileUploaderDropzoneInstructions"] {display: none;}</style>',
    unsafe_allow_html=True)
f = st.file_uploader('FASTA file', type=['fasta', 'fa', 'fna', 'txt'])
if st.button('Predict'):
    st.session_state.pop('records', None)
    try:
        if f is not None and seq.strip():
            raise ValueError('Use either pasted sequence or FASTA upload, not both.')
        records = parse_fasta(f.getvalue().decode('utf-8-sig')) if f is not None else [('pasted_sequence', clean_sequence(seq))]
        st.session_state['records'] = records
    except (ValueError, UnicodeDecodeError) as error:
        st.error(str(error))

# Preserve scores across slider reruns, so changing the threshold changes the verdict.
records = st.session_state.get('records')
if records:
    sequences = [record[1] for record in records]
    counts_all = vec.transform(sequences).toarray()
    gc = np.array([round(100 * (s.count('G') + s.count('C')) / len(s), 2)
                   for s in sequences]).reshape(-1, 1)
    X_input = pd.DataFrame(np.hstack((counts_all, gc)), columns=model['columns'])
    probabilities = clf.predict_proba(X_input)[:, 1]
    results = pd.DataFrame({'Record': [r[0] for r in records],
                            'Length': [len(s) for s in sequences],
                            'Probability': probabilities,
                            'Threshold': thr,
                            'Predicted promoter': probabilities >= thr})
    st.download_button('Download predictions', results.to_csv(index=False), 'predictions.csv', 'text/csv')
    for selected, (name, sequence) in enumerate(records):
        verdict = 'Predicted promoter' if probabilities[selected] >= thr else 'Not predicted'
        st.markdown(
            f"Length: {len(sequence)} bases  \n"
            f"Probability: {probabilities[selected]:.4f}  \n"
            f"Threshold: {thr:.2f}  \n"
            f"{verdict}")
        p = float(probabilities[selected])
        st.metric('Probability', f'{p:.4f}')
        st.progress(p)
        if p >= thr:
            st.success(f'Predicted promoter (threshold {thr:.2f})')
        else:
            st.info(f'Not predicted (threshold {thr:.2f})')
        counts = counts_all[selected]
        row = X_input.iloc[selected].to_numpy(dtype=np.float32)
        # Explanation is calculated at app runtime and does not change the classifier.
        contributions = np.zeros(clf.n_features_in_)
        for estimator in clf.estimators_:
            tree = estimator.tree_
            values = tree.value[:, 0, :]
            probability = values[:, 1] / values.sum(axis=1)
            node = 0
            while tree.children_left[node] != tree.children_right[node]:
                feature = tree.feature[node]
                child = tree.children_left[node] if row[feature] <= tree.threshold[node] else tree.children_right[node]
                contributions[feature] += probability[child] - probability[node]
                node = child
        contributions /= len(clf.estimators_)
        # In a forest, absence (count 0) can also drive a decision-path contribution.
        top = np.argsort(np.abs(contributions[:len(counts)]))[-10:][::-1]
        st.subheader('Top k-mers driving this prediction')
        table = pd.DataFrame({'k-mer': vec.get_feature_names_out()[top], 'Count': counts[top],
                               'Contribution': contributions[top]})
        st.dataframe(table, hide_index=True)
        st.bar_chart(table.set_index('k-mer')['Contribution'])
