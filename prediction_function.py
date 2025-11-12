import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
# from torch_geometric.data import Data
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv
import warnings

# Ignore specific warnings from scikit-learn
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Define the GCN model class (must match the trained model)
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.dropout1 = Dropout(0.5)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout2 = Dropout(0.5)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.dropout3 = Dropout(0.5)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.dropout1(x)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.dropout2(x)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.dropout3(x)
        out = self.lin(x)
        return out


def predict_fraud(new_data_df, model_path="gcn_correlation_smote_model.pth", correlation_threshold=0.5):
    """
    Predict fraud for new transaction data using the trained GCN model
    with a correlation-based graph.

    Args:
        new_data_df (pd.DataFrame): DataFrame containing the new transaction data.
                                    Must have the same columns as the training data (excluding 'Class').
        model_path (str): Path to the saved model state dictionary.
                          Defaults to "gcn_correlation_smote_model.pth".
        correlation_threshold (float): Correlation threshold used for graph construction.
                                       Defaults to 0.5.

    Returns:
        pd.DataFrame: DataFrame with original data, predicted class labels (0/1),
                      and prediction probabilities for the fraud class.
                      Returns None if model loading fails.
    """
    print("Starting fraud prediction...")

    processed_data_df = new_data_df.copy()

    expected_columns = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    if not all(col in processed_data_df.columns for col in expected_columns):
        print("Error: Input DataFrame does not have the expected columns.")
        return None

    # --- Preprocessing ---
    data_for_scaling = processed_data_df.copy()
    for col in data_for_scaling.columns:
        # Calculate IQR only if the column exists in the dataframe
        if col in data_for_scaling.columns:
            Q1 = data_for_scaling[col].quantile(0.25)
            Q3 = data_for_scaling[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data_for_scaling[col] = np.where(data_for_scaling[col] < lower_bound, lower_bound, data_for_scaling[col])
            data_for_scaling[col] = np.where(data_for_scaling[col] > upper_bound, upper_bound, data_for_scaling[col])


    # Scale features (using a new scaler fitted only on the new data for simplicity)
    scaler = StandardScaler()
    X_new_scaled = scaler.fit_transform(data_for_scaling)
    X_new_scaled = pd.DataFrame(X_new_scaled, columns=data_for_scaling.columns)

    # --- Drop 'Time' column before graph construction and model initialization ---
    if 'Time' in X_new_scaled.columns:
        X_new_scaled = X_new_scaled.drop('Time', axis=1)


    # --- Graph Construction  ---
    # Calculate the correlation matrix between features of the new data
    # Use a smaller subset for correlation calculation if the dataset is very large
    subset_for_corr = X_new_scaled.sample(n=min(10000, len(X_new_scaled)), random_state=42) if len(X_new_scaled) > 10000 else X_new_scaled
    correlation_matrix_new = subset_for_corr.corr()


    # Create adjacency matrix using the correlation threshold
    adj_matrix_correlation_new = (np.abs(correlation_matrix_new) > correlation_threshold).astype(int)

    # Remove self-loops (diagonal elements)
    np.fill_diagonal(adj_matrix_correlation_new.values, 0)

    # Convert adjacency matrix to a sparse representation
    adj_matrix_sparse_new = csr_matrix(adj_matrix_correlation_new)

    edge_index_new = torch.tensor(adj_matrix_sparse_new.nonzero(), dtype=torch.long)

    # Convert features to PyTorch tensor
    x_new_tensor = torch.tensor(X_new_scaled.values, dtype=torch.float)

    # Create PyTorch Geometric Data object
    data_new = Data(x=x_new_tensor, edge_index=edge_index_new)

    # --- Model Loading and Prediction ---
    # Check if CUDA is available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_node_features = X_new_scaled.shape[1]
    hidden_channels = 64 # Based on previous experiments
    num_classes = 2 # Binary classification (Fraud/Non-Fraud)

    model = GCN(num_node_features=num_node_features, hidden_channels=hidden_channels, num_classes=num_classes).to(device)

    # Load the saved model state dictionary
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set the model to evaluation mode
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please ensure the model was saved correctly.")
        # Attempt to load from Colab files if in Colab environment and file not found locally
        try:
            from google.colab import files
            print(f"Attempting to load from Colab files: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        except ImportError:
             print("Not in Google Colab environment. Please ensure the model file is accessible at the specified path.")
             return None
        except FileNotFoundError:
             print(f"Model file still not found after attempting to load from Colab files: {model_path}")
             return None
        except Exception as e:
             print(f"Error loading model from Colab files: {e}")
             return None
    except Exception as e:
        print(f"Error loading model state dictionary: {e}")
        return None

    # Perform inference
    with torch.no_grad():
        # Move data to the same device as the model
        data_new = data_new.to(device)
        out = model(data_new.x, data_new.edge_index)
        probabilities = F.softmax(out, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    print("Fraud prediction finished.")

    # Add predictions and probabilities back to the original DataFrame
    processed_data_df['Predicted_Class'] = predictions.cpu().numpy()
    processed_data_df['Fraud_Probability'] = probabilities.cpu().numpy()[:, 1]


    return processed_data_df
