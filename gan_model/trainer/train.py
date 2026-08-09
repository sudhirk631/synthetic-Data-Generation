import argparse
import os
import tempfile
import subprocess
#from google.cloud import bigquery
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
import joblib
import pandas as pd
from google.cloud import storage
import zipfile

# ============================================================================
# CONFIGURATION: Global hyperparameters for GAN training
# ============================================================================
NOISE_DIM = 128          # Dimension of random noise input to generator
BATCH_SIZE = 64          # Number of samples per training batch
LAMBDA = 10              # Gradient penalty weight for WGAN
NUM_CLASSES = 1          # Number of output classes (not used in current implementation)

# ============================================================================
# STEP 1: CLOUD STORAGE - GCS Upload Function
# Uploads generated models and synthetic data to Google Cloud Storage
# ============================================================================
def upload_to_gcs(local_path, bucket_name, destination_blob_name):
    """Upload a local file to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)

# ============================================================================
# STEP 2: DATA TRANSFORMATION - NullTransformer Class
# Handles missing values in data by filling with mean/mode/custom value
# and optionally tracking null positions for reconstruction
# ============================================================================
class NullTransformer:
    """Transformer for data that contains Null values."""
    nulls = None
    _null_column = None
    _fill_value = None

    def __init__(self, fill_value, null_column=None, copy=False):
        """Initialize transformer with fill strategy."""
        self.fill_value = fill_value
        self.null_column = null_column
        self.copy = copy

    def fit(self, data):
        """Fit the transformer to the data - analyze null patterns."""
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        null_values = data.isnull()
        self.nulls = null_values.any()
        contains_not_null = not null_values.all()

        # Determine fill value strategy
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

# ============================================================================
# STEP 3: DATA TRANSFORMATION - DatetimeTransformer Class
# Converts datetime strings to normalized numeric values for GAN training
# and handles reverse transformation back to datetime format
# ============================================================================
class DatetimeTransformer:
    """Transformer for datetime data."""
    null_transformer = None
    divider = None
 
    def __init__(self, nan='mean', null_column=False, strip_constant=False):
        """Initialize datetime transformer with null handling strategy."""
        self.nan = nan
        self.null_column = null_column
        self.strip_constant = strip_constant

    def _find_divider(self, transformed):
        """Find optimal divider for datetime values to reduce dimensionality."""
        self.divider = 1
        multipliers = [10] * 9 + [60, 60, 24]
        for multiplier in multipliers:
            candidate = self.divider * multiplier
            if np.mod(transformed, candidate).any():
                break
            self.divider = candidate

    def _transform(self, datetimes):
        """Transform datetime values to numeric (nanoseconds) representation."""
        datetimes_str = datetimes.astype(str)
        # Try multiple datetime formats for flexibility
        parsed = pd.to_datetime(datetimes_str, errors='coerce', infer_datetime_format=True)
        parsed = parsed.fillna(pd.to_datetime(datetimes_str, format='%H:%M:%S', errors='coerce'))
        parsed = parsed.fillna(pd.to_datetime(datetimes_str, format='%H:%M', errors='coerce'))
        # Convert to nanoseconds as float
        integers = parsed.values.astype('datetime64[ns]').astype(np.float64)
        integers[pd.isnull(parsed)] = np.nan
        transformed = pd.Series(integers)

        if self.strip_constant:
            self._find_divider(transformed.fillna(0))
            transformed = transformed.floordiv(self.divider)

        return transformed

    def fit(self, data):
        """Fit transformer - calculate min/max for normalization."""
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        
        transformed = self._transform(data)
        self._min = transformed.min()
        range_val = transformed.max() - transformed.min()
        self._scale = range_val if range_val != 0 else 1

        # Setup null handling for transformed datetime data
        self.null_transformer = NullTransformer(self.nan, self.null_column, copy=True)
        scaled = (transformed - self._min) / self._scale
        print(scaled)
        self.null_transformer.fit(scaled)
        self.num_dt_cols = self.null_transformer.num_dt_cols

    def transform(self, data):
        """Transform datetime data to normalized numeric values."""
        if isinstance(data, np.ndarray):
            data = pd.Series(data.reshape(-1))

        data = self._transform(data)
        data_scaled = (data - self._min) / self._scale
        data_scaled = data_scaled.fillna(np.nan)

        return self.null_transformer.transform(data_scaled)

    def reverse_transform(self, data):
        """Convert normalized numeric values back to datetime format."""
        if self.nan is not None:
            data = self.null_transformer.reverse_transform(data)

        if isinstance(data, np.ndarray) and data.ndim == 2:
            data = data[:, 0]

        # Reverse normalization and conversion
        data = data.astype(np.float64) * self._scale + self._min

        if self.strip_constant:
            data = data * self.divider

        return pd.to_datetime(data)

# ============================================================================
# STEP 4: DATA TRANSFORMATION - CategoricalTransformer Class
# Converts categorical values to continuous values using probability intervals
# Enables smooth gradient flow during GAN training
# ============================================================================
class CategoricalTransformer:
    """Transforms categorical data to continuous values for GAN training."""
    
    def __init__(self, fuzzy=False, clip=False):
        """Initialize with fuzzy (random) or deterministic transformation."""
        self.fuzzy = fuzzy
        self.clip = clip

    @staticmethod
    def _get_intervals(data):
        """Calculate probability-based intervals for each category."""
        frequencies = data.value_counts(dropna=False)
        start = 0
        intervals = {}
        means = []
        starts = []
        elements = len(data)

        # Assign each category a range [0,1] proportional to its frequency
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
        """Fit transformer - calculate intervals for all unique categories."""
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        self.mapping = {}
        self.dtype = data.dtype
        self.intervals, self.means, self.starts = self._get_intervals(data)
        self._get_category_from_index = list(self.means.index).__getitem__
        
    def _transform_by_category(self, data):
        """Transform categorical data using category-based lookup."""
        result = np.empty(len(data), dtype=float)

        for category, values in self.intervals.items():
            mean, std = values[2], values[3]

            if category is np.nan:
                mask = data.isnull()
            else:
                mask = data == category

            if self.fuzzy:
                # Add random variation from normal distribution
                result[mask] = norm.rvs(mean, std, size=mask.sum())
            else:
                result[mask] = mean

        return result

    def _get_value(self, category):
        """Get transformed value for a single category."""
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
        """Transform data row by row using category lookup."""
        return data.fillna(np.nan).apply(self._get_value).to_numpy()

    def transform(self, data):
        """Public transform method - convert categorical to continuous."""
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        return self._transform_by_row(data)

    def _normalize(self, data):
        """Normalize transformed values to [0,1] range."""
        if self.clip:
            return data.clip(0, 1)

        return np.mod(data, 1)

    def _reverse_transform_by_matrix(self, data):
        """Reverse transform using efficient matrix operations (high memory)."""
        num_rows = len(data)
        num_categories = len(self.means)
        data_expanded = np.broadcast_to(data, (num_categories, num_rows)).T
        means_expanded = np.broadcast_to(self.means.values, (num_rows, num_categories))
        diffs = np.abs(data_expanded - means_expanded)
        indexes = np.argmin(diffs, axis=1)
        self._get_category_from_index = list(self.means.index).__getitem__

        return pd.Series(indexes).apply(self._get_category_from_index).astype(self.dtype)

    def _get_category_from_start(self, value):
        """Map continuous value back to nearest category."""
        lower = self.starts.loc[:value]
        return lower.iloc[-1].category

    def _reverse_transform_by_row(self, data):
        """Reverse transform row by row (low memory)."""
        return data.apply(self._get_category_from_start).astype(self.dtype)

    def reverse_transform(self, data):
        """Convert continuous values back to original categories."""
        if not isinstance(data, pd.Series):
            if len(data.shape) > 1:
                data = data[:, 0]
            data = pd.Series(data)
        data = self._normalize(data)
        num_rows = len(data)
        num_categories = len(self.means)
        needed_memory = num_rows * num_categories * 8 * 3
        available_memory = psutil.virtual_memory().available
        
        # Choose method based on available system memory
        if available_memory > needed_memory:
            return self._reverse_transform_by_matrix(data)
        return self._reverse_transform_by_row(data)

# ============================================================================
# STEP 5: GAN ARCHITECTURE - Generator Network
# Creates synthetic data from random noise
# Uses separate branches for categorical and datetime features
# ============================================================================
def create_generator_combined(num_cat, num_dt):
    """Create generator network with separate paths for different data types."""
    noise_input = Input(shape=(NOISE_DIM,))
    
    # Categorical branch: transforms noise to categorical feature range
    cat_branch = Dense(128)(noise_input)
    cat_branch = LeakyReLU(0.2)(cat_branch)
    cat_branch = BatchNormalization()(cat_branch)
    cat_branch = Dense(num_cat, activation='sigmoid')(cat_branch)

    # Datetime branch: transforms noise to datetime feature range
    dt_branch = Dense(128)(noise_input)
    dt_branch = LeakyReLU(0.2)(dt_branch)
    dt_branch = BatchNormalization()(dt_branch)
    dt_branch = Dense(num_dt, activation='sigmoid')(dt_branch)

    # Concatenate both branches to create final output
    combined_output = Concatenate()([cat_branch, dt_branch])
    model = Model(inputs=noise_input, outputs=combined_output)

    return model

# ============================================================================
# STEP 6: GAN ARCHITECTURE - Critic Network (WGAN Discriminator)
# Evaluates how "real" synthetic data appears compared to real data
# ============================================================================
def create_critic_combined(num_cat, num_dt):
    """Create critic network to distinguish real from fake data."""
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

# ============================================================================
# STEP 7: LOSS FUNCTIONS - Critic and Generator Loss
# Uses Wasserstein loss for stable GAN training
# ============================================================================
def critic_loss(real_output, fake_output):
    """WGAN critic loss: maximize distance between real and fake."""
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def generator_loss(fake_output):
    """WGAN generator loss: minimize distance between fake and real."""
    return -tf.reduce_mean(fake_output)

# ============================================================================
# STEP 8: POST-PROCESSING - Decoding Functions
# Convert generated continuous values back to original data types
# ============================================================================
def decode_generated_dt(generated, dt_transformers, datetime_features):
    """Decode generated datetime features back to datetime format."""
    n_dt = len(datetime_features)
    dt_arrays = np.split(generated[:, :n_dt], n_dt, axis=1) if n_dt > 0 else []
    dt_decoded = []
    for arr, col in zip(dt_arrays, datetime_features):
        dt_decoded.append(dt_transformers[col].reverse_transform(arr.flatten()))

    data = {col: dt_decoded[i] for i, col in enumerate(datetime_features)}
    df = pd.DataFrame(data)

    return df

def decode_generated_cat(generated, cat_transformers, categorical_features):
    """Decode generated categorical features back to original categories."""
    n_cat = len(categorical_features)
    cat_arrays = np.split(generated[:, :n_cat], n_cat, axis=1) if n_cat > 0 else []
    cat_decoded = []
    for arr, col in zip(cat_arrays, categorical_features):
        cat_decoded.append(cat_transformers[col].reverse_transform(arr.flatten()))

    data = {col: cat_decoded[i] for i, col in enumerate(categorical_features)}
    df = pd.DataFrame(data)

    return df

# ============================================================================
# STEP 9: TRAINING UTILITY - Gradient Penalty
# Enforces Lipschitz constraint for WGAN stability
# ============================================================================
def gradient_penalty(real_data, generated_data, crit):
    """Calculate gradient penalty for WGAN critic regularization."""
    batch_size = tf.shape(real_data)[0]
    alpha = tf.random.uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)
    # Interpolate between real and generated data
    interpolated = real_data + alpha * (generated_data - real_data)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        critic_output = crit(interpolated, training=True)

    gradients = tape.gradient(critic_output, interpolated)
    gradients = tf.reshape(gradients, [batch_size, -1])
    grad_norm = tf.norm(gradients, axis=1)
    # Penalize gradients that deviate from norm of 1
    gp = tf.reduce_mean((grad_norm - 1.0) ** 2)

    return gp
    
# ============================================================================
# STEP 10: TRAINING LOOP - Single Training Step
# Updates critic 5x per generator update for stable WGAN training
# ============================================================================
def train_step(real_data, gen, crit, g_optimizer, d_optimizer):
    """Execute one training iteration: update critic then generator."""
    crit.trainable = True
    
    # CRITIC UPDATE: Train critic for 5 iterations per generator step
    for _ in range(5):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

        with tf.GradientTape() as tape:
            generated_data = gen(noise, training=True)
            # Add noise to real data for smoother critic training
            real_noisy = real_data + tf.random.normal(shape=tf.shape(real_data), mean=0.0, stddev=0.01)
            real_output = crit(real_noisy, training=True)
            fake_output = crit(generated_data, training=True)
            # Calculate gradient penalty for Lipschitz constraint
            gp = gradient_penalty(real_data, generated_data, crit)
            # WGAN loss with gradient penalty
            c_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + LAMBDA * gp

        c_gradients = tape.gradient(c_loss, crit.trainable_variables)
        d_optimizer.apply_gradients(zip(c_gradients, crit.trainable_variables))

    # GENERATOR UPDATE: Train generator once per training step
    crit.trainable = False
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as tape:
        generated_data = gen(noise, training=True)
        # Ensure output is in valid range [0, 1]
        generated_data = tf.clip_by_value(generated_data, 0.0, 1.0)
        fake_output = crit(generated_data, training=True)
        g_loss = -tf.reduce_mean(fake_output)

    g_gradients = tape.gradient(g_loss, gen.trainable_variables)
    g_optimizer.apply_gradients(zip(g_gradients, gen.trainable_variables))

    return c_loss, g_loss, gp

# ============================================================================
# STEP 11: INFERENCE - Generate Synthetic Data
# Sample from generator to create synthetic samples
# ============================================================================
def generate_synthetic(generator, num_samples):
    """Generate synthetic data samples using trained generator."""
    noise = np.random.normal(0, 1, (num_samples, NOISE_DIM))
    generated = generator.predict(noise, verbose=0)

    return generated  # Returns data scaled between 0-1

# ============================================================================
# STEP 12: DECODING - Convert Synthetic Data to Original Format
# Reverse all transformations to match original data format
# ============================================================================
def decode_synthetic(synthetic, num_cat_features, cat_transformers, categorical_features,
                     dt_transformers, datetime_features, original_data, high_null):
    """Convert generated normalized data back to original data types and format."""
    
    # Split into categorical and datetime portions
    synthetic_cat = synthetic[:, :num_cat_features]
    synthetic_dt = synthetic[:, num_cat_features:]

    # Decode categorical features
    cat_arrays = np.split(synthetic_cat, len(categorical_features), axis=1)
    cat_decoded = []

    for arr, col in zip(cat_arrays, categorical_features):
        cat_decoded.append(cat_transformers[col].reverse_transform(arr.flatten()))

    # Decode datetime features
    dt_arrays = np.split(synthetic_dt, len(datetime_features), axis=1)
    dt_decoded = []
    for arr, col in zip(dt_arrays, datetime_features):
        decoded = dt_transformers[col].reverse_transform(arr.flatten())
        dt_decoded.append(decoded)

    # Create DataFrames for each feature type
    df_cat = pd.DataFrame({col: cat_decoded[i] for i, col in enumerate(categorical_features)})
    df_dt = pd.DataFrame({col: dt_decoded[i] for i, col in enumerate(datetime_features)})
    # Keep original high-null features as-is (not generated)
    nulls = pd.DataFrame({col: original_data[col] for col in high_null})

    # Combine all feature types into single DataFrame
    return pd.concat([df_cat, df_dt, nulls], axis=1)

# ============================================================================
# STEP 13: PREPROCESSING & TRAINING - Main Training Function
# Orchestrates data loading, transformation, model creation, and training loop
# ============================================================================
                         
def preprocess_and_train(original_data, steps):
    """Main preprocessing and training pipeline."""
    
    # Remove time-specific columns that shouldn't be learned
    time_cols_to_drop = [col for col in original_data.columns if col.lower().endswith('_time')]
    original_data.drop(columns=time_cols_to_drop, inplace=True, errors='ignore')
    
    # PHASE 1: Detect and convert datetime columns
    date_features = []
    for col in original_data.columns:
        if pd.api.types.is_datetime64_any_dtype(original_data[col]):
            continue
        sample = original_data[col].dropna().head(1000).astype(str)
        if sample.empty:
            continue
        date_features.append(col)
        parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
        # If > 80% of values parse as datetime, treat as datetime column
        if parsed.notna().mean() > 0.8:
            original_data[col] = pd.to_datetime(original_data[col], errors='coerce', infer_datetime_format=True)
            # Standardize format as string for consistency
            original_data[col] = original_data[col].dt.strftime('%Y-%m-%d %H:%M:%S')

    # PHASE 2: Identify feature types
    # Columns with >70% nulls are treated as high-null (not generated)
    high_null = [col for col in original_data.columns if original_data[col].isnull().mean() > 0.7]
    # Remaining columns are treated as categorical features
    categorical_features = [col for col in original_data.columns if col not in date_features and col not in high_null]
    # Datetime columns that aren't high-null
    datetime_features = [col for col in date_features if col in original_data.columns and col not in high_null]

    # PHASE 3: Fit and transform categorical features
    print("Fitting categorical transformers...")
    cat_transformers = {col: CategoricalTransformer() for col in categorical_features}
    for col, tr in cat_transformers.items():
        tr.fit(original_data[col])

    # Transform categorical data to continuous values
    cat_data = np.column_stack([tr.transform(original_data[col]) for col, tr in cat_transformers.items()])
    # Save transformers for later inference
    joblib.dump(cat_transformers, 'cat_transformers.pkl')
    
    # PHASE 4: Fit and transform datetime features
    print("Fitting datetime transformers...")
    dt_transformers = {col: DatetimeTransformer() for col in datetime_features}
    for col, tr in dt_transformers.items():
        tr.fit(original_data[col])

    # Save transformers for later inference
    joblib.dump(dt_transformers, 'dt_transformers.pkl')

    # Transform datetime data to continuous values
    dt_data = np.column_stack([tr.transform(original_data[col]) for col, tr in dt_transformers.items()])
    
    # PHASE 5: Normalize all features to [0, 1] range
    print("Normalizing features to [0, 1] range...")
    cat_scaler = MinMaxScaler()
    cat_real = cat_scaler.fit_transform(cat_data)
    dt_scaler = MinMaxScaler()
    dt_real = dt_scaler.fit_transform(dt_data)
    
    # Get dimensions for GAN architecture
    NUM_DT = dt_real.shape[1]
    NUM_CAT = cat_real.shape[1]
    
    # Combine categorical and datetime features
    combined_real_data = np.hstack([cat_real, dt_real])
    df1 = pd.DataFrame(combined_real_data)
    
    # PHASE 6: Create GAN models
    print("Creating GAN models...")
    gen = create_generator_combined(NUM_CAT, NUM_DT)
    crit = create_critic_combined(NUM_CAT, NUM_DT)
    
    # PHASE 7: Setup optimizers with appropriate learning rates
    g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    d_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    
    # Convert data to TensorFlow format
    real_data = tf.cast(combined_real_data, tf.float32)

    # PHASE 8: Training loop - run for specified number of steps
    print(f"Starting training for {steps} steps...")
    for step in range(steps):
        # Sample random batch from real data
        idx = np.random.randint(0, combined_real_data.shape[0], BATCH_SIZE)
        real_batch = tf.gather(real_data, idx)
        # Execute training step
        c_loss, g_loss, gp_val = train_step(real_batch, gen, crit, g_optimizer, d_optimizer)

        # Log progress every 100 steps
        if step % 100 == 0:
            print(f"Step: {step}, Critic Loss: {c_loss.numpy()}, Generator Loss: {g_loss.numpy()}, GP: {gp_val.numpy()}")

    # PHASE 9: Generate synthetic data after training completes
    print("Generating synthetic data...")
    synthetic_data = generate_synthetic(gen, 1000)

    print("Synthetic data shape:", synthetic_data.shape)

    # PHASE 10: Decode synthetic data back to original format
    print("Decoding synthetic data...")
    num_cat_features = cat_real.shape[1]
    synthetic_df = decode_synthetic(synthetic_data, num_cat_features, cat_transformers, categorical_features, 
                                    dt_transformers, datetime_features, original_data, high_null)

    print(synthetic_df.head())

    return synthetic_df, gen

# ============================================================================
# STEP 14: MAIN EXECUTION - Orchestrates entire pipeline
# Loads data, trains model, generates synthetic data, saves outputs
# ============================================================================

def main(args):
    """Main entry point: load data, train GAN, generate and save synthetic data."""
    
    # Load original training data from CSV
    print(f"Loading data from {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    
    # Run full preprocessing and training pipeline
    print("Starting preprocessing and training...")
    synthetic_df, gen = preprocess_and_train(df, args.steps)
    
    # Align columns between original and synthetic data
    df = df[[col for col in df.columns if col in synthetic_df.columns]]
   
    # SAVE PHASE 1: Save trained generator model to GCS
    print("Saving trained generator model...")
    export_path = args.model_dir
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_export_path = os.path.join(tmpdirname, "gan_model")
        gen.export(local_export_path)
        
        # Upload the uncompressed model directory to GCS
        bucket_name = export_path.split('/')[2]
        destination_blob_prefix = '/'.join(export_path.split('/')[3:])

        # Upload all model files to GCS
        for root, dirs, files in os.walk(local_export_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Compute relative path inside the model directory
                rel_path = os.path.relpath(file_path, local_export_path)
                gcs_blob_name = destination_blob_prefix + '/' + rel_path
                upload_to_gcs(file_path, bucket_name, gcs_blob_name)

    print(f"Model saved to {export_path}")
    
    # SAVE PHASE 2: Save synthetic data CSV to GCS
    print("Saving synthetic data to CSV...")
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_export_path = os.path.join(tmpdirname, "gan_model")
        synthetic_df.to_csv(local_export_path, index=False)

        # Upload the CSV to GCS
        bucket_name = args.output_path.split('/')[2]
        destination_blob_name = '/'.join(args.output_path.split('/')[3:])
        upload_to_gcs(local_export_path, bucket_name, destination_blob_name)

    print(f"Synthetic data written to {args.output_path}")

    # EVALUATION PHASE: Generate quality report comparing synthetic to real data
    print("Generating quality report...")
    feature_schema = {
        "primary_key": None,
        "columns": {
            "passenger_id": {"sdtype":  "categorical"},
            "passenger_id_hash": {"sdtype": "categorical"},
            "passenger_version_sequence":  {"sdtype": "categorical"},
            "number_of_passenger": {"sdtype": "categorical"},
            "number_of_infant": {"sdtype": "categorical"},
            "channel_id": {"sdtype": "categorical"},
            "number_of_air_segment": {"sdtype": "categorical"},
            "agency_name":  {"sdtype": "categorical"},
            "year": {"sdtype": "categorical"},
            "month": {"sdtype": "categorical"},
            "day": {"sdtype": "categorical"},
            "primary_traveler_flag": {"sdtype":  "categorical"},
            "passenger_sequence": {"sdtype": "categorical"},
            "origin_destination_seq":  {"sdtype": "categorical"},
            "flight_number": {"sdtype": "categorical"},
            "airline_cd": {"sdtype": "categorical"},
            "action_cd": {"sdtype": "categorical"},
            "service_end_city": {"sdtype": "categorical"},
            "service_start_city": {"sdtype": "categorical"},
            "class_of_service": {"sdtype": "categorical"},
            "dayofweek_indicator": {"sdtype": "categorical"},
            "seat_category":  {"sdtype": "categorical"},
            "departure_airport": {"sdtype": "categorical"},
            "arrival_airport": {"sdtype": "categorical"},
            "departure_at_airport_ts": {"sdtype": "datetime"},
            "arrival_datetime": {"sdtype": "datetime"},
            "departure_datetime":{"sdtype": "datetime"},
            "cabin_category": {"sdtype": "categorical"},
            "frequent_traveler_number": {"sdtype": "categorical"},
        }
    }

    report = QualityReport()
    report.generate(df, synthetic_df, feature_schema)

    # Quality report shows metrics like Column Shapes, Semantic Similarity, etc.
    # Evaluating Column Shapes: |██████████| 124/124 [00:01<00:00, 114.90it/s]|
    # Column Shapes Score: 74.03%
    # Modify the Data Preparation Stage if accuracy goes down

# ============================================================================
# STEP 15: SCRIPT ENTRY POINT - Parse arguments and run main()
# ============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/input.csv",
        help="Path to input CSV file for training data"
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

    # Execute main pipeline
    main(args)

# ============================================================================
# SETUP NOTES & USAGE INSTRUCTIONS
# ============================================================================
# Setup requirement: Configure Docker authentication for GCS
# Run once: gcloud auth configure-docker us-docker.pkg.dev
#
# To run this script, use the following bash command:
# python train.py --csv-path "path/to/your/data.csv" --steps 2000
#
# Optional arguments:
#   --csv-path: Path to input CSV file (default: data/input.csv)
#   --model-dir: GCS path for saving model (default: gs://gan_test_1/tmp/gan_model_direct)
#   --output-path: GCS path for synthetic CSV (default: gs://gan_test_1/final_generated_direct.csv)
#   --steps: Number of training iterations (default: 2000)
# ============================================================================
