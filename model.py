import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import optuna

class SpaceshipDataset(Dataset):
    def __init__(self, X, y=None, device='cpu'):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.y is not None:
            return self.X[index], self.y[index]
        return self.X[index]

class SpaceshipMLP(nn.Module):
    def __init__(self, input_dim, n_layers, n_units, dropout_rate):
        super().__init__()
        layers = []
        in_features = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, n_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = n_units
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def preprocess_data(train_df, test_df):
    train_size = len(train_df)
    y = train_df['Transported'].astype(float).values
    df = pd.concat([train_df.drop(['Transported'], axis=1), test_df], axis=0).reset_index(drop=True)
    
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    group_size = df['Group'].value_counts().to_dict()
    df['GroupSize'] = df['Group'].map(group_size)
    
    df['CabinDeck'] = df['Cabin'].apply(lambda x: str(x).split('/')[0] if pd.notnull(x) else np.nan)
    df['CabinSide'] = df['Cabin'].apply(lambda x: str(x).split('/')[2] if pd.notnull(x) else np.nan)
    
    expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in expense_cols:
        df[col] = df[col].fillna(0)
    df['TotalExpense'] = df[expense_cols].sum(axis=1)
    
    df = df.drop(['PassengerId', 'Group', 'Cabin', 'Name'], axis=1)
    
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'CabinDeck', 'CabinSide']
    num_cols = ['Age', 'GroupSize', 'TotalExpense'] + expense_cols
    
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    num_imputer = SimpleImputer(strategy='median')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    
    for col in expense_cols + ['TotalExpense']:
        df[col] = np.log1p(df[col])
        
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = ohe.fit_transform(df[cat_cols])
    
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(df[num_cols])
    
    X_processed = np.hstack([num_scaled, cat_encoded])
    
    X_train = X_processed[:train_size]
    X_test = X_processed[train_size:]
    
    return X_train, y, X_test

def objective(trial, X, y, device):
    n_layers = trial.suggest_int('n_layers', 1, 4)
    n_units = trial.suggest_int('n_units', 16, 128, step=16)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    epochs = 30
    batch_size = 128

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        train_dataset = SpaceshipDataset(X_tr, y_tr, device)
        val_dataset = SpaceshipDataset(X_val, y_val, device)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = SpaceshipMLP(input_dim=X.shape[1], n_layers=n_layers, n_units=n_units, dropout_rate=dropout_rate).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):
            model.train()
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        model.eval()
        val_preds = []
        val_targets_list = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
                val_preds.extend(preds)
                val_targets_list.extend(targets.cpu().numpy())

        acc = accuracy_score(val_targets_list, val_preds)
        fold_accuracies.append(acc)

    return np.mean(fold_accuracies)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_csv('datasets/spaceship-titanic/train.csv')
    test_df = pd.read_csv('datasets/spaceship-titanic/test.csv')

    X_train, y_train, X_test = preprocess_data(train_df, test_df)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print("Starting Optuna tuning (30 trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, device), n_trials=30)

    print("Best hyperparameters: ", study.best_params)
    print(f"Best cross-validation accuracy: {study.best_value:.4f}")

    best_params = study.best_params
    batch_size = 128
    epochs = 50

    print("Training final model on full dataset...")
    final_model = SpaceshipMLP(
        input_dim=X_train.shape[1],
        n_layers=best_params['n_layers'],
        n_units=best_params['n_units'],
        dropout_rate=best_params['dropout_rate']
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

    train_dataset = SpaceshipDataset(X_train, y_train, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    final_model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    print("Predicting on test data...")
    test_dataset = SpaceshipDataset(X_test, device=device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    final_model.eval()
    test_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            outputs = final_model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).bool().cpu().numpy()
            test_preds.extend(preds)

    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': [p[0] for p in test_preds]
    })
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to 'submission.csv'")