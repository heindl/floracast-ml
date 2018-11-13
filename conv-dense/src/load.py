"""
Used to load datasets from the data directory.
"""

import numpy as np

from helper import loaders

class Loader(object):
    loaders = {
            'libras': loaders.load_libras,
            'arabic': loaders.load_arabic_digits,
            'trajectories': loaders.load_trajectories,
            'ecg': loaders.load_ecg,
            'wafer': loaders.load_wafer,
            'auslan': loaders.load_auslan,
            'beef': loaders.load_uni_csv,
            'yoga': loaders.load_uni_csv,
            'coffee': loaders.load_uni_csv,
            'ecg200': loaders.load_uni_csv,
            'faces': loaders.load_uni_csv,
            'cricket_y': loaders.load_uni_csv,
            'lighting_2': loaders.load_uni_csv,
            'two_lead_ecg': loaders.load_uni_csv,
            'mote_strain': loaders.load_uni_csv,
            'two_patterns': loaders.load_uni_csv,
    }

    def __init__(self, correct_dims=None):
        self.correct_dims = correct_dims
        pass

    def load(self, data_dir, datasets):
        data_dict = {}

        for k,v in datasets:
            print 'Loading %s' % k
            data_dict[k] = Loader.loaders[k](data_dir, v)
            assert (self.has_correct_form(data_dict[k])), 'Data set "%s" does not have correct form' % k
            print 'Done loading %s' % k

        return data_dict

    def has_correct_form(self, data):
        # Check that correct number of items returned
        if len(data)==2:
            X_tr,y_tr = data

            unique_tr = np.unique(y_tr)
            if np.any(unique_tr != np.arange(len(unique_tr))): return False
        elif len(data)==4:
            X_tr,y_tr,X_te,y_te = data

            unique_tr = np.unique(y_tr)
            if np.any(unique_tr != np.arange(len(unique_tr))): return False

            unique_te = np.unique(y_te)
            if np.any(unique_te != np.arange(len(unique_te))): return False
        else:
            return False

        # Check data dims if specified
        if self.correct_dims:
            correct_tr,matched_dims = self.has_correct_dims(X_tr)
            correct_te,_ = self.has_correct_dims(X_te, matched_dims) if len(data)==4 else (True,None)

            if not (correct_tr and correct_te):
                return False

        return True

    def has_correct_dims(self, X, matched_dims=None):
        dims = []
        try:
            for i,d in enumerate(self.correct_dims):
                dims.append(len(X))

                if d == '?':
                    X = X[0]
                elif d == '$':
                    # Should also match dims from other data matrices from this dataset
                    if matched_dims and len(X) != matched_dims[i]:
                        return False, None
                    X = X[0]
                else:
                    if len(X) != d: 
                        return False, None
        except Exception:
            return False, None

        return True, dims

