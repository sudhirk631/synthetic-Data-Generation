import argparse
import os
import tempfile
import subprocess
from google.cloud import bigquery
import numpy as np
import datetime
from sdmetrics.reports.single_table import QualityReport
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop,Adam
import psutil
from scipy.stats import norm
from google.cloud import bigquery
import joblib
import pandas as pd
from google.cloud import storage
import zipfile

NOISE_DIM = 128
BATCH_SIZE = 64
LAMBDA = 10
NUM_CLASSES = 1

def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)

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
        """Fit the transformer to the data."""
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

        if self.null_column is None:
            self._null_column = self.nulls
        else:
            self._null_column = self.null_column

        self.num_dt_cols = 2 if self._null_column else 1

    def transform(self, data):
        """Replace null values with the indicated fill_value and optionally create null indicator column."""
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        if self.nulls:
            isnull = data.isnull()
            if self._fill_value is not None:
                if not self.copy:
                    data.loc[isnull] = self._fill_value
                else:
                    data = data.fillna(self._fill_value)

            if self._null_column:
                return pd.concat([data, isnull.astype(int)], axis=1).values

        return data.values.reshape(-1, 1)

    def reverse_transform(self, data):
        """Revert the transform, replacing fill values back to nulls."""

        if self.nulls:

            if self._null_column:

                isnull_mask = data[:, 1] > 0.5

                data_col = pd.Series(data[:, 0])

            else:

                data_col = pd.Series(data.ravel())

                # Replace fill values with NaN using tolerance for floats

                if pd.api.types.is_numeric_dtype(data_col):

                    isnull_mask = np.isclose(data_col.values, self._fill_value, atol=1e-8)

                else:

                    isnull_mask = data_col == self._fill_value

 

            if isnull_mask.any():

                if self.copy:

                    data_col = data_col.copy()

                data_col.loc[isnull_mask] = np.nan

            return data_col

 

        # If no nulls detected during fit, return data as Series

        return pd.Series(data.ravel())

 

 

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

        multipliers = [10] * 9 + [60, 60, 24]

        for multiplier in multipliers:

            candidate = self.divider * multiplier

            if np.mod(transformed, candidate).any():

                break

            self.divider = candidate

 

    def _transform(self, datetimes):

        """Transform datetime values to integer (nanoseconds)."""

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

        print(transformed)

        self._min = transformed.min()

        range_val = transformed.max() - transformed.min()

        self._scale = range_val if range_val != 0 else 1

        # nanoseconds in a day (adjust for ns)

        self.null_transformer = NullTransformer(self.nan, self.null_column, copy=True)

        scaled = (transformed - self._min) / self._scale

        print(scaled)

        self.null_transformer.fit(scaled)

        self.num_dt_cols = self.null_transformer.num_dt_cols

 

    def transform(self, data):

        if isinstance(data, np.ndarray):

            data = pd.Series(data.reshape(-1))

        data = self._transform(data)

        data_scaled = (data - self._min) / self._scale

        data_scaled = data_scaled.fillna(np.nan)

        return self.null_transformer.transform(data_scaled)

 

    def reverse_transform(self, data):

        if self.nan is not None:

            data = self.null_transformer.reverse_transform(data)

        if isinstance(data, np.ndarray) and data.ndim == 2:

            data = data[:, 0]

        data = data.astype(np.float64) * self._scale + self._min

        if self.strip_constant:

            data = data * self.divider

        return pd.to_datetime(data)

 

class CategoricalTransformer:

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

        self.mapping = {}

        self.dtype = data.dtype

        self.intervals, self.means, self.starts = self._get_intervals(data)

        self._get_category_from_index = list(self.means.index).__getitem__

 

    def _transform_by_category(self, data):

        result = np.empty(len(data), dtype=float)

        for category, values in self.intervals.items():

            mean, std = values[2], values[3]

            if category is np.nan:

                mask = data.isnull()

            else:

                mask = data == category

            if self.fuzzy:

                result[mask] = norm.rvs(mean, std, size=mask.sum())

            else:

                result[mask] = mean

        return result

 

    def _get_value(self, category):

        if pd.isnull(category):

            category = np.nan

        else:

            key_types = list(self.intervals.keys())

            if key_types:

                key_type = type(key_types[0])

                # Only cast if both are numbers, otherwise use as is

                try:

                    if (isinstance(category, (int, float, np.number)) and

                            issubclass(key_type, (int, float, np.number))):

                        category = key_type(category)

                except Exception:

                    pass

        if category not in self.intervals:

            raise KeyError(f"Category {category} not found in intervals keys: {list(self.intervals.keys())}")

        mean, std = self.intervals[category][2:4]

        if self.fuzzy:

            return norm.rvs(mean, std)

        return mean

 

    def _transform_by_row(self, data):

        return data.fillna(np.nan).apply(self._get_value).to_numpy()

 

    def transform(self, data):

        if not isinstance(data, pd.Series):

            data = pd.Series(data)

        return self._transform_by_row(data)

 

    def _normalize(self, data):

        if self.clip:

            return data.clip(0, 1)

        return np.mod(data, 1)

 

    def _reverse_transform_by_matrix(self, data):

        num_rows = len(data)

        num_categories = len(self.means)

        data_expanded = np.broadcast_to(data, (num_categories, num_rows)).T

        means_expanded = np.broadcast_to(self.means.values, (num_rows, num_categories))

        diffs = np.abs(data_expanded - means_expanded)

        indexes = np.argmin(diffs, axis=1)

        self._get_category_from_index = list(self.means.index).__getitem__

        return pd.Series(indexes).apply(self._get_category_from_index).astype(self.dtype)

 

    def _get_category_from_start(self, value):

        lower = self.starts.loc[:value]

        return lower.iloc[-1].category

 

    def _reverse_transform_by_row(self, data):

        return data.apply(self._get_category_from_start).astype(self.dtype)

 

    def reverse_transform(self, data):

        if not isinstance(data, pd.Series):

            if len(data.shape) > 1:

                data = data[:, 0]

            data = pd.Series(data)

        data = self._normalize(data)

        num_rows = len(data)

        num_categories = len(self.means)

        needed_memory = num_rows * num_categories * 8 * 3

        available_memory = psutil.virtual_memory().available

        if available_memory > needed_memory:

            return self._reverse_transform_by_matrix(data)

        return self._reverse_transform_by_row(data)

 

def create_generator_combined(num_cat, num_dt):

    noise_input = Input(shape=(NOISE_DIM,))

 

    # Categorical branch

    cat_branch = Dense(128)(noise_input)

    cat_branch = LeakyReLU(0.2)(cat_branch)

    cat_branch = BatchNormalization()(cat_branch) #add nodes

    cat_branch = Dense(num_cat, activation='sigmoid')(cat_branch)

 

    # Datetime branch

    dt_branch = Dense(128)(noise_input)

    dt_branch = LeakyReLU(0.2)(dt_branch)

    dt_branch = BatchNormalization()(dt_branch)

    dt_branch = Dense(num_dt, activation='sigmoid')(dt_branch)

 

    # Concatenate both branches

    combined_output = Concatenate()([cat_branch, dt_branch])

 

    model = Model(inputs=noise_input, outputs=combined_output)

    return model

 

def create_critic_combined(num_cat, num_dt):

    data_input = Input(shape=(num_cat + num_dt,))

    x = Dense(256)(data_input)

    x = LeakyReLU(0.2)(x)

    x = Dropout(0.3)(x)

    x = Dense(128)(x)

    x = LeakyReLU(0.2)(x)

    x = Dropout(0.3)(x)

    output = Dense(1)(x)

    model = Model(inputs=data_input, outputs=output)

    return model

 

def critic_loss(real_output, fake_output):

    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

 

 

def generator_loss(fake_output):

    return -tf.reduce_mean(fake_output)

 

 

 

 

def decode_generated_dt(generated, dt_transformers, datetime_features):

    n_dt = len(datetime_features)

    dt_arrays = np.split(generated[:, :n_dt], n_dt, axis=1) if n_dt > 0 else []

    dt_decoded = []

    for arr, col in zip(dt_arrays, datetime_features):

        dt_decoded.append(dt_transformers[col].reverse_transform(arr.flatten()))

    data = {col: dt_decoded[i] for i, col in enumerate(datetime_features)}

    df = pd.DataFrame(data)

    return df

 

 

 

def decode_generated_cat(generated, cat_transformers, categorical_features):

    n_cat = len(categorical_features)

    cat_arrays = np.split(generated[:, :n_cat], n_cat, axis=1) if n_cat > 0 else []

    cat_decoded = []

    for arr, col in zip(cat_arrays, categorical_features):

        cat_decoded.append(cat_transformers[col].reverse_transform(arr.flatten()))

    data = {col: cat_decoded[i] for i, col in enumerate(categorical_features)}

    df = pd.DataFrame(data)

    return df

 

 

def gradient_penalty(real_data, generated_data,crit):

    batch_size = tf.shape(real_data)[0]

    alpha = tf.random.uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)

    interpolated = real_data + alpha * (generated_data - real_data)

    with tf.GradientTape() as tape:

        tape.watch(interpolated)

        critic_output = crit(interpolated, training=True)

    gradients = tape.gradient(critic_output, interpolated)

    gradients = tf.reshape(gradients, [batch_size, -1])

    grad_norm = tf.norm(gradients, axis=1)

    gp = tf.reduce_mean((grad_norm - 1.0) ** 2)

    return gp

 

def train_step(real_data,gen,crit,g_optimizer,d_optimizer):

    crit.trainable = True

    for _ in range(5):  # critic updates

        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

        with tf.GradientTape() as tape:

            generated_data = gen(noise, training=True)

            real_noisy = real_data + tf.random.normal(shape=tf.shape(real_data), mean=0.0, stddev=0.01)

            real_output = crit(real_noisy, training=True)

            fake_output = crit(generated_data, training=True)

            gp = gradient_penalty(real_data, generated_data,crit)

            c_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + LAMBDA * gp

        c_gradients = tape.gradient(c_loss, crit.trainable_variables)

        d_optimizer.apply_gradients(zip(c_gradients, crit.trainable_variables))

    # Generator update

    crit.trainable = False

    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as tape:

        generated_data = gen(noise, training=True)

        generated_data = tf.clip_by_value(generated_data, 0.0, 1.0)

        fake_output = crit(generated_data, training=True)

        g_loss = -tf.reduce_mean(fake_output)

    g_gradients = tape.gradient(g_loss, gen.trainable_variables)

    g_optimizer.apply_gradients(zip(g_gradients, gen.trainable_variables))

    return c_loss, g_loss, gp

 

def generate_synthetic(generator, num_samples):

    noise = np.random.normal(0, 1, (num_samples, NOISE_DIM))

    generated = generator.predict(noise, verbose=0)

    return generated  # scaled between 0-1

 

def decode_synthetic(synthetic, num_cat_features, cat_transformers, categorical_features,

                     dt_transformers, datetime_features, original_data,high_null):

    synthetic_cat = synthetic[:, :num_cat_features]

    synthetic_dt = synthetic[:, num_cat_features:]

 

    # Decode categorical features

    cat_arrays = np.split(synthetic_cat, len(categorical_features), axis=1)

    cat_decoded = []

    for arr, col in zip(cat_arrays, categorical_features):

        cat_decoded.append(cat_transformers[col].reverse_transform(arr.flatten()))

    dt_arrays = np.split(synthetic_dt, len(datetime_features), axis=1)

    dt_decoded = []

    for arr, col in zip(dt_arrays, datetime_features):

        decoded = dt_transformers[col].reverse_transform(arr.flatten())

        # Format each datetime string to remove fractional seconds

        dt_decoded.append(decoded)

    # Create DataFrames

    df_cat = pd.DataFrame({col: cat_decoded[i] for i, col in enumerate(categorical_features)})

    df_dt = pd.DataFrame({col: dt_decoded[i] for i, col in enumerate(datetime_features)})

    nulls = pd.DataFrame({col: original_data[col] for col in high_null})

    return pd.concat([df_cat, df_dt,nulls], axis=1)

 

def preprocess_and_train(original_data, steps):

    time_cols_to_drop = [col for col in original_data.columns if col.lower().endswith('_time')]

    original_data.drop(columns=time_cols_to_drop, inplace=True, errors='ignore')

    date_features = []

    for col in original_data.columns:

        if pd.api.types.is_datetime64_any_dtype(original_data[col]):

            continue

        sample = original_data[col].dropna().head(1000).astype(str)

        if sample.empty:

            continue

        date_features.append(col)

        parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)

        if parsed.notna().mean() > 0.8:

            original_data[col] = pd.to_datetime(original_data[col], errors='coerce', infer_datetime_format=True)

            # Standardize format as string

            original_data[col] = original_data[col].dt.strftime('%Y-%m-%d %H:%M:%S')

 

    # --- Feature Setup ---

    high_null = [col for col in original_data.columns if original_data[col].isnull().mean() > 0.7]

 

    categorical_features = [col for col in original_data.columns if col not in date_features and col not in high_null]

    datetime_features = [col for col in date_features if col in original_data.columns and col not in high_null]

 

    #Transformers to convert cat and datetime fields

    cat_transformers = {col: CategoricalTransformer() for col in categorical_features}

    for col, tr in cat_transformers.items():

        tr.fit(original_data[col])

    cat_data = np.column_stack([tr.transform(original_data[col]) for col, tr in cat_transformers.items()])

    joblib.dump(cat_transformers, 'cat_transformers.pkl')

    # Fit datetime transformers per column

    dt_transformers = {col: DatetimeTransformer() for col in datetime_features}

    for col, tr in dt_transformers.items():

        tr.fit(original_data[col])

    joblib.dump(dt_transformers, 'dt_transformers.pkl')

 

    # Transform

    dt_data = np.column_stack([tr.transform(original_data[col]) for col, tr in dt_transformers.items()])

 

    cat_scaler = MinMaxScaler()

    cat_real = cat_scaler.fit_transform(cat_data)

    dt_scaler = MinMaxScaler()

    dt_real = dt_scaler.fit_transform(dt_data)

    NUM_DT = dt_real.shape[1]

    NUM_CAT = cat_real.shape[1]

    combined_real_data = np.hstack([cat_real, dt_real])

    df1=pd.DataFrame(combined_real_data)

 

    gen = create_generator_combined(NUM_CAT, NUM_DT)

    crit = create_critic_combined(NUM_CAT, NUM_DT)

    g_optimizer = Adam(learning_rate=0.0002,beta_1=0.5)

    d_optimizer = Adam(learning_rate=0.0002,beta_1=0.5)

 

    real_data = tf.cast(combined_real_data, tf.float32)

    for step in range(steps):

        idx = np.random.randint(0, combined_real_data.shape[0], BATCH_SIZE)

        real_batch = tf.gather(real_data, idx)

        c_loss, g_loss, gp_val = train_step(real_batch,gen,crit,g_optimizer,d_optimizer)

        if step % 100 == 0:

            print(f"Step: {step}, Critic Loss: {c_loss.numpy()}, Generator Loss: {g_loss.numpy()}, GP: {gp_val.numpy()}")

 

    synthetic_data = generate_synthetic(gen, 1000)

    print("Synthetic data shape:", synthetic_data.shape)

 

    num_cat_features = cat_real.shape[1]

    synthetic_df = decode_synthetic(synthetic_data, num_cat_features, cat_transformers, categorical_features, dt_transformers, datetime_features,original_data,high_null)

    print(synthetic_df.head())

    return synthetic_df,gen

 

 

 

def main(args):

    # Fetch data from bq

    client = bigquery.Client(project="sab-dev-dap-aimlpipeline-4474")

    df = client.query(args.query).to_dataframe()

 

    # Runing preprocessing + training

    synthetic_df,gen = preprocess_and_train(df, args.steps)

    df = df[[col for col in df.columns if col in synthetic_df.columns]]

    # Save the trained generator for serving

    export_path = args.model_dir

    with tempfile.TemporaryDirectory() as tmpdirname:

        local_export_path = os.path.join(tmpdirname, "gan_model")

        gen.export(local_export_path)

        # Upload the uncompressed model directory to GCS

        bucket_name = export_path.split('/')[2]

        destination_blob_prefix = '/'.join(export_path.split('/')[3:])

        for root, dirs, files in os.walk(local_export_path):

            for file in files:

                file_path = os.path.join(root, file)

                # Compute relative path inside the model directory

                rel_path = os.path.relpath(file_path, local_export_path)

                gcs_blob_name = destination_blob_prefix + '/' + rel_path

                upload_to_gcs(file_path, bucket_name, gcs_blob_name)

    print(f" Model saved to {export_path}")

    with tempfile.TemporaryDirectory() as tmpdirname:

        local_export_path = os.path.join(tmpdirname, "gan_model")

        synthetic_df.to_csv(local_export_path, index=False)

        # Upload the CSV

        bucket_name = args.output_path.split('/')[2]

        destination_blob_name = '/'.join(args.output_path.split('/')[3:])

        upload_to_gcs(local_export_path, bucket_name, destination_blob_name)

    print(f" Synthetic data written to {args.output_path}")

    feature_schema = {

        "primary_key": None,

        "columns": {

            "pnr_locator_id": {"sdtype":  "categorical"},

            "pnr_locator_id_hash": {"sdtype": "categorical"},

            "pnr_sequence":  {"sdtype": "categorical"},

            "number_in_party": {"sdtype": "categorical"},

            "number_of_infant": {"sdtype": "categorical"},

            "source_system_id": {"sdtype": "categorical"},

            "number_of_air_segment": {"sdtype": "categorical"},

            "pcc":  {"sdtype": "categorical"},

            "year": {"sdtype": "categorical"},

            "month": {"sdtype": "categorical"},

            "day": {"sdtype": "categorical"},

            "unique_id": {"sdtype": "categorical"},

            "passenger_name_hash": {"sdtype": "categorical"},

            "primary_traveler_flag": {"sdtype":  "categorical"},

            "name_id":  {"sdtype": "categorical"},

            "name_association_id": {"sdtype": "categorical"},

            "name_type": {"sdtype": "categorical"},

            "traveller_seq_nbr": {"sdtype": "categorical"},

            "passenger_sequence": {"sdtype": "categorical"},

            "unique_id_1": {"sdtype": "categorical"},

            "segment_association_id": {"sdtype": "categorical"},

            "od_seq":  {"sdtype": "categorical"},

            "tripmarket_origin":  {"sdtype": "categorical"},

            "tripmarket_destination": {"sdtype": "categorical"},

            "active_segment_flag": {"sdtype": "categorical"},

            "segment_number": {"sdtype": "categorical"},

            "ordered_segment_number": {"sdtype": "categorical"},

            "segment_type_cd": {"sdtype": "categorical"},

            "flight_number": {"sdtype": "categorical"},

            "operating_airline_cd": {"sdtype": "categorical"},

            "operating_flight_number": {"sdtype": "categorical"},

            "airline_cd": {"sdtype": "categorical"},

            "action_cd": {"sdtype": "categorical"},

            "service_end_city": {"sdtype": "categorical"},

            "service_start_city": {"sdtype": "categorical"},

            "number_in_party_1": {"sdtype": "categorical"},

            "equipment_type":  {"sdtype": "categorical"},

            "equipment_cd": {"sdtype": "categorical"},

            "marketing_airline_cd": {"sdtype": "categorical"},

            "marketing_flight_number": {"sdtype": "categorical"},

            "operating_class_of_service": {"sdtype": "categorical"},

            "marketing_class_of_service": {"sdtype": "categorical"},

            "dayofweek_indicator": {"sdtype": "categorical"},

            "pos_agency_city_cd": {"sdtype": "categorical"},

            "pos_country_cd": {"sdtype": "categorical"},

            "ancillary_id": {"sdtype": "categorical"},

            "pdc_seat":  {"sdtype": "categorical"},

            "rfic_cd": {"sdtype": "categorical"},

            "rfic_subcode": {"sdtype": "categorical"},

            "name_association_id_1": {"sdtype": "categorical"},

            "pax_type": {"sdtype": "categorical"},

            "segment_indicator": {"sdtype": "categorical"},

            "number_of_items": {"sdtype": "categorical"},

            "associated_segment_count": {"sdtype": "categorical"},

            "status_indicator": {"sdtype": "categorical"},

            "travel_portion":  {"sdtype": "categorical"},

            "group_cd": {"sdtype": "categorical"},

            "board_point": {"sdtype": "categorical"},

            "off_point": {"sdtype": "categorical"},

            "ttl_price":  {"sdtype": "categorical"},

            "commercial_name": {"sdtype": "categorical"},

            "airline_cd_1": {"sdtype": "categorical"},

            "flight_number_1": {"sdtype": "categorical"},

            "class_of_service":  {"sdtype": "categorical"},

            "tvp_unique_id": {"sdtype": "categorical"},

            "ae_portion_sequence": {"sdtype": "categorical"},

            "tvp_airline_cd": {"sdtype": "categorical"},

            "tvp_flight_number": {"sdtype":  "categorical"},

            "tvp_operating_flight_number": {"sdtype": "categorical"},

            "tvp_class_of_service": {"sdtype": "categorical"},

            "tvp_board_point": {"sdtype": "categorical"},

            "tvp_off_point": {"sdtype":  "categorical"},

            "tvp_marketing_carrier":  {"sdtype": "categorical"},

            "tvp_operating_carrier": {"sdtype": "categorical"},

            "tvp_segment_association_id": {"sdtype": "categorical"},

            "service_end_dt": {"sdtype": "datetime"},

            "service_start_dt": {"sdtype": "datetime"},

            "departure_datetime": {"sdtype": "datetime"},

            "segment_booked_ts": {"sdtype": "datetime"},

            "segment_booked_dt": {"sdtype": "datetime"},

            "departure_at_airport_ts": {"sdtype": "datetime"},

            "arrival_datetime": {"sdtype": "datetime"},

            "tvp_departure_dt":  {"sdtype": "datetime"},

            "departure_dt": {"sdtype": "datetime"},

            "ingest_ts": {"sdtype": "datetime"},

            "load_ts": {"sdtype": "datetime"},

            "pnr_create_dt": {"sdtype": "datetime"},

            "transmission_ts": {"sdtype": "datetime"},

            "from_ts": {"sdtype": "datetime"},

            "pnr_create_ts": {"sdtype": "datetime"},

            "tty_airline_cd": {"sdtype":  "categorical"},

            "oac_accounting_cd": {"sdtype": "categorical"},

            "passenger_type": {"sdtype": "categorical"},

            "frequent_flyer_id": {"sdtype":  "categorical"},

            "ff_number": {"sdtype": "categorical"},

            "frequent_flyer_tier_level": {"sdtype": "categorical"},

            "ff_level_type": {"sdtype": "categorical"},

            "ff_tier_level_number": {"sdtype": "categorical"},

            "ff_supplier_cd": {"sdtype": "categorical"},

            "agent_iata_number": {"sdtype": "categorical"},

            "tier_level": {"sdtype": "categorical"},

            "segment_group": {"sdtype": "categorical"},

            "cabin_cd": {"sdtype": "categorical"},

            "vendor_name": {"sdtype": "categorical"},

            "supplier_name": {"sdtype": "categorical"},

            "departure_ts": {"sdtype": "datetime"},

            "arrival_ts": {"sdtype": "datetime"},

            "frequent_traveler_number": {"sdtype": "categorical"},

            "offer_snap_id": {"sdtype": "categorical"},

            "frequent_flyer_tier": {"sdtype": "categorical"},

            "eticket_coupon":  {"sdtype": "categorical"},

            "eticket_number":  {"sdtype": "categorical"},

            "emd_coupon":  {"sdtype": "categorical"},

            "emd_number": {"sdtype": "categorical"},

            "bag_weight": {"sdtype": "categorical"},

            "ttl_price_currency": {"sdtype": "categorical"},

            "booking_indicator": {"sdtype": "categorical"},

            "purchase_datetime": {"sdtype": "datetime"},

            "purchase_ts": {"sdtype": "datetime"},

            "offer_id": {"sdtype": "categorical"},

            "offer_item_id": {"sdtype": "categorical"},

            "offer_source": {"sdtype": "categorical"},

            "sequence_number": {"sdtype": "categorical"},

            "equipment_type_1": {"sdtype": "categorical"}

        }

    }

    report = QualityReport()

    report.generate(df, synthetic_df, feature_schema)

    # Evaluating Column Shapes: |██████████| 124/124 [00:01<00:00, 114.90it/s]|

    # Column Shapes Score: 74.03%

    # Modify the Data Preparation Stage if accuracy goes down

 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--query",

        type=str,

        default="SELECT * FROM `sab-dev-dap-aimlpipeline-4474.sampledomain_aiml.test_gan_pnr` ",

        help="BigQuery SQL query for training data"

    )

    parser.add_argument(

        "--model-dir",

        type=str,

        default="gs://gan_test_1/tmp/gan_model_direct",

        help="Path for saving trained model"

    )

    parser.add_argument(

        "--output-path",

        type=str,

        default="gs://gan_test_1/final_generated_direct.csv",

        help="Destination GCS path for synthetic CSV"

    )

    parser.add_argument(

        "--steps",

        type=int,

        default=2000,

        help="Number of training steps"

    )

    args = parser.parse_args()

    main(args)

#fix for auth error was to run gcloud auth configure-docker us-docker.pkg.dev"""

