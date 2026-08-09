"""Data transformers for encoding/decoding categorical and datetime features."""

import numpy as np

import pandas as pd

import psutil

from scipy.stats import norm

 

 

class NullTransformer:

    """Transformer for data that contains Null values."""

    nulls = None

    _null_column = None

    _fill_value = None

 

    def __init__(self, fill_value, null_column=None, copy=False):

        self.fill_value = fill_value

        self.null_column = null_column

        self.copy = copy

 

    def fit(self, data):

        if not isinstance(data, pd.Series):

            data = pd.Series(data)

        null_values = data.isnull()

        self.nulls = null_values.any()

        contains_not_null = not null_values.all()

        if self.fill_value == 'mean':

            self._fill_value = data.mean() if contains_not_null else 0

        elif self.fill_value == 'mode':

            mode_vals = data.mode(dropna=True)

            self._fill_value = mode_vals[0] if not mode_vals.empty else 0

        else:

            self._fill_value = self.fill_value

        self._null_column = self.nulls if self.null_column is None else self.null_column

        self.num_dt_cols = 2 if self._null_column else 1

 

    def transform(self, data):

        if not isinstance(data, pd.Series):

            data = pd.Series(data)

        if self.nulls:

            isnull = data.isnull()

            if self._fill_value is not None:

                data = data.fillna(self._fill_value) if self.copy else data.copy().fillna(self._fill_value)

            if self._null_column:

                return pd.concat([data, isnull.astype(int)], axis=1).values

        return data.values.reshape(-1, 1)

 

    def reverse_transform(self, data):

        if self.nulls:

            if self._null_column:

                isnull_mask = data[:, 1] > 0.5

                data_col = pd.Series(data[:, 0])

            else:

                data_col = pd.Series(data.ravel())

                if pd.api.types.is_numeric_dtype(data_col):

                    isnull_mask = np.isclose(data_col.values, self._fill_value, atol=1e-8)

                else:

                    isnull_mask = data_col == self._fill_value

            if isnull_mask.any():

                data_col = data_col.copy()

                data_col.loc[isnull_mask] = np.nan

            return data_col

        return pd.Series(data.ravel())

 

 

class CategoricalTransformer:

    """Transformer for categorical features."""

 

    def __init__(self, fuzzy=False, clip=False):

        self.fuzzy = fuzzy

        self.clip = clip

 

    @staticmethod

    def _get_intervals(data):

        frequencies = data.value_counts(dropna=False)

        start = 0

        intervals = {}

        means = []

        starts = []

        elements = len(data)

        for value, frequency in frequencies.items():

            prob = frequency / elements

            end = start + prob

            mean = start + prob / 2

            std = prob / 6

            val = np.nan if pd.isnull(value) else value

            intervals[val] = (start, end, mean, std)

            means.append(mean)

            starts.append((val, start))

            start = end

        means = pd.Series(means, index=list(frequencies.keys()))

        starts_df = pd.DataFrame(starts, columns=['category', 'start']).set_index('start')

        return intervals, means, starts_df

 

    def fit(self, data):

        if isinstance(data, np.ndarray):

            data = pd.Series(data)

        self.dtype = data.dtype

        self.intervals, self.means, self.starts = self._get_intervals(data)

        self._get_category_from_index = list(self.means.index).__getitem__

 

    def _get_value(self, category):

        if pd.isnull(category):

            category = np.nan

        if category not in self.intervals:

            raise KeyError(f"Category {category} not found in intervals.")

        mean, std = self.intervals[category][2:4]

        return norm.rvs(mean, std) if self.fuzzy else mean

 

    def transform(self, data):

        if not isinstance(data, pd.Series):

            data = pd.Series(data)

        return data.fillna(np.nan).apply(self._get_value).to_numpy()

 

    def _normalize(self, data):

        return data.clip(0, 1) if self.clip else np.mod(data, 1)

 

    def _reverse_transform_by_matrix(self, data):

        num_rows = len(data)

        num_categories = len(self.means)

        data_expanded = np.broadcast_to(data, (num_categories, num_rows)).T

        means_expanded = np.broadcast_to(self.means.values, (num_rows, num_categories))

        indexes = np.argmin(np.abs(data_expanded - means_expanded), axis=1)

        self._get_category_from_index = list(self.means.index).__getitem__

        return pd.Series(indexes).apply(self._get_category_from_index).astype(self.dtype)

 

    def _get_category_from_start(self, value):

        return self.starts.loc[:value].iloc[-1].category

 

    def reverse_transform(self, data):

        if not isinstance(data, pd.Series):

            if len(data.shape) > 1:

                data = data[:, 0]

            data = pd.Series(data)

        data = self._normalize(data)

        needed_memory = len(data) * len(self.means) * 8 * 3

        if psutil.virtual_memory().available > needed_memory:

            return self._reverse_transform_by_matrix(data)

        return data.apply(self._get_category_from_start).astype(self.dtype)

 

 

class DatetimeTransformer:

    """Transformer for datetime data."""

    null_transformer = None

    divider = None

 

    def __init__(self, nan='mean', null_column=False, strip_constant=False):

        self.nan = nan

        self.null_column = null_column

        self.strip_constant = strip_constant

 

    def _find_divider(self, transformed):

        self.divider = 1

        for multiplier in [10] * 9 + [60, 60, 24]:

            candidate = self.divider * multiplier

            if np.mod(transformed, candidate).any():

                break

            self.divider = candidate

 

    def _transform(self, datetimes):

        datetimes_str = datetimes.astype(str)

        parsed = pd.to_datetime(datetimes_str, errors='coerce', infer_datetime_format=True)

        parsed = parsed.fillna(pd.to_datetime(datetimes_str, format='%H:%M:%S', errors='coerce'))

        parsed = parsed.fillna(pd.to_datetime(datetimes_str, format='%H:%M', errors='coerce'))

        integers = parsed.values.astype('datetime64[ns]').astype(np.float64)

        integers[pd.isnull(parsed)] = np.nan

        transformed = pd.Series(integers)

        if self.strip_constant:

            self._find_divider(transformed.fillna(0))

            transformed = transformed.floordiv(self.divider)

        return transformed

 

    def fit(self, data):

        if isinstance(data, np.ndarray):

            data = pd.Series(data)

        transformed = self._transform(data)

        self._min = transformed.min()

        range_val = transformed.max() - transformed.min()

        self._scale = range_val if range_val != 0 else 1

        self.null_transformer = NullTransformer(self.nan, self.null_column, copy=True)

        self.null_transformer.fit((transformed - self._min) / self._scale)

        self.num_dt_cols = self.null_transformer.num_dt_cols

 

    def transform(self, data):

        if isinstance(data, np.ndarray):

            data = pd.Series(data.reshape(-1))

        data = self._transform(data)

        data_scaled = (data - self._min) / self._scale

        return self.null_transformer.transform(data_scaled.fillna(np.nan))

 

    def reverse_transform(self, data):

        if self.nan is not None:

            data = self.null_transformer.reverse_transform(data)

        if isinstance(data, np.ndarray) and data.ndim == 2:

            data = data[:, 0]

        data = data.astype(np.float64) * self._scale + self._min

        if self.strip_constant:

            data = data * self.divider

        return pd.to_datetime(data)

 

 

def decode_synthetic(synthetic, num_cat_features, cat_transformers, categorical_features, dt_transformers, datetime_features):

    """Decodes synthetic GAN output back to original feature space."""

    synthetic = np.array(synthetic)

    synthetic_cat = synthetic[:, :num_cat_features]

    synthetic_dt = synthetic[:, num_cat_features:]

 

    cat_decoded = [

        cat_transformers[col].reverse_transform(arr.flatten())

        for arr, col in zip(np.split(synthetic_cat, len(categorical_features), axis=1), categorical_features)

    ]

    dt_decoded = [

        dt_transformers[col].reverse_transform(arr.flatten())

        for arr, col in zip(np.split(synthetic_dt, len(datetime_features), axis=1), datetime_features)

    ]

 

    df_cat = pd.DataFrame({col: cat_decoded[i] for i, col in enumerate(categorical_features)})

    df_dt = pd.DataFrame({col: dt_decoded[i] for i, col in enumerate(datetime_features)})

    return pd.concat([df_cat, df_dt], axis=1)
