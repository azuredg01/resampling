import argparse
import pandas as pd
import os
import ast
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, SMOTENC, SMOTEN

"""
example usage:
# ["RandomOverSampler", "SMOTE", "ADASYN", "BorderlineSMOTE", "SVMSMOTE", "KMeansSMOTE"]
# numeric data
python oversampling_v1.py -m SMOTE -in numeric_data.csv -out SMOTE.csv -ss {0: 100, 1: 150}

# ["SMOTEN"]
# categorical data
python oversampling_v1.py -m SMOTEN -in categorical_data.csv -out SMOTEN.csv -ss {0: 100, 1: 150}

# ["SMOTENC"] + -cf
# mixed data
python oversampling_v1.py -m SMOTENC -in mixed_data.csv -out SMOTENC.csv -cf 1 4 -ss {0: 100, 1: 150}

Input dataset must be 2D: shape (n_samples, n_features)
"""
parser = argparse.ArgumentParser(description="OverSampling")
parser.add_argument("-m", "--method", type=str, required=True,
                    choices=["RandomOverSampler", "SMOTE", "ADASYN", "BorderlineSMOTE",
                                "SVMSMOTE", "KMeansSMOTE", "SMOTENC", "SMOTEN"],
                    help="Over-sampling method name")
parser.add_argument("-in", "--input", type=str, required=True, help="Input npy file")
parser.add_argument("-out", "--output", type=str, required=True, help="Output npy file")
parser.add_argument("-ss", "--sampling_strategy", type=lambda s: s if s.lower() == "auto" else ast.literal_eval(s),
                    default="auto",
                    help="Sampling strategy: 'auto', float (e.g. 0.5), int (e.g. 100), or dict (e.g. '{0: 100, 1: 300}')")
parser.add_argument("-cf", "--categorical_features", type=int, nargs='+', default=None, help="for SMOTENC only, e.g., -cf 1 4")

def run_random_oversampler(x, y, sampling_strategy="auto"):
    sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42, shrinkage=None)
    return sampler.fit_resample(x, y)

def run_smote(x, y, sampling_strategy="auto"):
    sampler = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=5, random_state=42)
    return sampler.fit_resample(x, y)

def run_adasyn(x, y, sampling_strategy="auto"):
    sampler = ADASYN(sampling_strategy=sampling_strategy, n_neighbors=5, random_state=42)
    return sampler.fit_resample(x, y)

def run_borderline_smote(x, y, sampling_strategy="auto"):
    sampler = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5, m_neighbors=10)
    return sampler.fit_resample(x, y)

def run_svmsmote(x, y, sampling_strategy="auto"):
    sampler = SVMSMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5, m_neighbors=10)
    return sampler.fit_resample(x, y)

def run_kmeans_smote(x, y, sampling_strategy="auto"):
    sampler = KMeansSMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=2)
    return sampler.fit_resample(x, y)

def run_smote_nc(x, y, cf, sampling_strategy="auto"):
    sampler = SMOTENC(categorical_features=cf, sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
    return sampler.fit_resample(x, y)

def run_smoten(x, y, sampling_strategy="auto"):
    sampler = SMOTEN(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
    return sampler.fit_resample(x, y)

def main():
    # input dataset must be 2D: shape (n_samples, n_features)
    data = pd.read_csv(args.input) # <------------------------------
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    ss = args.sampling_strategy
    cf = args.categorical_features
    method_func = {
        "RandomOverSampler": lambda x, y: run_random_oversampler(x, y, ss),
        "SMOTE": lambda x, y: run_smote(x, y, ss),
        "ADASYN": lambda x, y: run_adasyn(x, y, ss),
        "BorderlineSMOTE": lambda x, y: run_borderline_smote(x, y, ss),
        "SVMSMOTE": lambda x, y: run_svmsmote(x, y, ss),
        "KMeansSMOTE": lambda x, y: run_kmeans_smote(x, y, ss),
        "SMOTENC": lambda x, y: run_smote_nc(x, y, cf, ss),
        "SMOTEN": lambda x, y: run_smoten(x, y, ss)
    }

    if args.method == "SMOTENC" and args.categorical_features is None:
        raise ValueError("categorical_features must be provided for SMOTENC.")

    x_resampled, y_resampled = method_func[args.method](x, y)
    resampled_df = pd.DataFrame(x_resampled)# <------------------------------
    resampled_df["label"] = y_resampled# <------------------------------
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    resampled_df.to_csv(args.output, index=False) # <------------------------------
    print(f"Resampled data saved to {args.output}")

if __name__ == "__main__":
    args = parser.parse_args()
    main()