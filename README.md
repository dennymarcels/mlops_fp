# ML Classifier - Breast Cancer Prediction

A machine learning project that builds and deploys a neural network classifier for breast cancer prediction using the scikit-learn breast cancer dataset.

## Project Structure

```
├── app/                          # Web application
│   ├── main.py                   # Flask app with prediction API
│   └── templates/
│       └── index.html            # Web interface for predictions
├── src/                          # Source code modules
│   ├── data_loading/
│   │   └── load_data.py          # Dataset loading and preparation
│   ├── data_preprocessing/
│   │   └── preprocess_data.py    # Data cleaning and imputation
│   ├── feature_engineering/
│   │   └── engineer_features.py # Feature scaling and transformation
│   ├── model_training/
│   │   └── train_model.py        # Neural network training
│   └── model_evaluation/
│       └── evaluate_model.py     # Model performance evaluation
├── artifacts/                    # Trained model and preprocessing artifacts
├── data/                         # Data storage
│   ├── raw/                      # Raw dataset
│   ├── preprocessed/             # Cleaned data
│   └── processed/                # Feature-engineered data
├── metrics/                      # Model performance metrics
├── params.yaml                   # Configuration parameters
└── pyproject.toml               # Python dependencies
```

## Features

- **Data Pipeline**: Complete ETL pipeline from raw data to model-ready features
- **Neural Network**: TensorFlow/Keras deep learning model with configurable architecture
- **Web Interface**: Flask-based web application for making predictions
- **Artifact Management**: Serialized models and preprocessors for deployment
- **Evaluation Metrics**: Comprehensive model performance analysis

## Dependencies

The project requires Python 3.12+ and the packages informed in `pyproject.toml`.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd iaexp
```

2. Install dependencies:
```bash
pip install -e .
```

## Configuration

Model hyperparameters and data processing settings are configured in `params.yaml`.

## Model Architecture

The neural network consists of a multilayer perceptron with 2 hidden layers.

## Artifacts

The training process generates the following artifacts in the `artifacts/` directory:
- `model.keras`: Trained TensorFlow model
- `[features]_mean_imputer.joblib`: Feature imputer for missing values
- `[features]_scaler.joblib`: Standard scaler for feature normalization
- `[target]_one_hot_encoder.joblib`: One-hot encoder for target labels

## Metrics

Model performance metrics are saved to:
- `metrics/training.json`: Training history and validation metrics
- `metrics/evaluation.json`: Test set performance and confusion matrix

## Development

The project follows a modular structure with separate concerns:
- **Data Loading**: Fetches and saves raw breast cancer dataset
- **Preprocessing**: Handles missing values and data splitting
- **Feature Engineering**: Applies scaling transformations
- **Model Training**: Builds and trains the neural network
- **Model Evaluation**: Generates performance metrics
- **Web Application**: Provides prediction interface

Each module can be run independently and saves its outputs for the next stage in the pipeline.

## Usage

### Training the Model

Run the complete ML pipeline (for proper logging to the terminal, run as modules with `python -m`):

```bash
# 1. Load and prepare raw data
python -m src.data_loading.load_data

# 2. Preprocess data (imputation, train/test split)
python -m src.data_preprocessing.preprocess_data

# 3. Engineer features (scaling)
python -m src.feature_engineering.engineer_features

# 4. Train the neural network model
python -m src.model_training.train_model

# 5. Evaluate model performance
python -m src.model_evaluation.evaluate_model
```

### Running the Web Application

After training the model, start the Flask web server:

```bash
python app/main.py
```

The application will be available at `http://localhost:5001`

### Making Predictions

1. **Web Interface**: Upload a CSV file with breast cancer features through the web interface
2. **API**: The `/upload` endpoint accepts CSV files and returns predictions

#### Required CSV Format

Your CSV file must contain all 30 breast cancer features with exact column names:
- mean radius, mean texture, mean perimeter, mean area, mean smoothness, etc.
- See `sklearn.datasets.load_breast_cancer().feature_names` for the complete list
