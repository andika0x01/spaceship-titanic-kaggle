import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import optuna

def preprocess_data(train_df, test_df):
    train_size = len(train_df)
    y = train_df['Transported'].astype(int).values
    df = pd.concat([train_df.drop(['Transported'], axis=1), test_df], axis=0).reset_index(drop=True)
    
    # --- 1. Basic Splits ---
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    group_size = df['Group'].value_counts().to_dict()
    df['GroupSize'] = df['Group'].map(group_size)
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)
    
    df['Surname'] = df['Name'].apply(lambda x: str(x).split(' ')[-1] if pd.notnull(x) else np.nan)
    
    df['CabinDeck'] = df['Cabin'].apply(lambda x: str(x).split('/')[0] if pd.notnull(x) else np.nan)
    df['CabinNum'] = df['Cabin'].apply(lambda x: float(str(x).split('/')[1]) if pd.notnull(x) else np.nan)
    df['CabinSide'] = df['Cabin'].apply(lambda x: str(x).split('/')[2] if pd.notnull(x) else np.nan)

    # --- 2. Smart Imputation: Expenses & CryoSleep ---
    expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # If CryoSleep is True, expenses are 0
    for col in expense_cols:
        df.loc[df['CryoSleep'] == True, col] = 0
        df[col] = df[col].fillna(0)
        
    df['TotalExpense'] = df[expense_cols].sum(axis=1)
    df['HasExpense'] = (df['TotalExpense'] > 0).astype(int)
    
    # If spent money, cannot be in CryoSleep
    df.loc[df['TotalExpense'] > 0, 'CryoSleep'] = False
    # If missing CryoSleep and 0 expense, likely True
    df['CryoSleep'] = df['CryoSleep'].fillna(True)
    df['CryoSleep'] = df['CryoSleep'].astype(int)
    
    # --- 3. Smart Imputation: HomePlanet ---
    # Earthlings don't go to deck A, B, C, T. Only Europa.
    df.loc[(df['HomePlanet'].isna()) & (df['CabinDeck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet'] = 'Europa'
    # Deck G is only Earth
    df.loc[(df['HomePlanet'].isna()) & (df['CabinDeck'] == 'G'), 'HomePlanet'] = 'Earth'
    
    # By Group (Same Group = Same Planet)
    df['HomePlanet'] = df['HomePlanet'].fillna(df.groupby('Group')['HomePlanet'].transform('first'))
    # By Surname (Surnames are usually linked to Planet)
    df['HomePlanet'] = df['HomePlanet'].fillna(df.groupby('Surname')['HomePlanet'].transform('first'))
    df['HomePlanet'] = df['HomePlanet'].fillna('Earth')
    
    # --- 4. Smart Imputation: Cabin ---
    df['CabinSide'] = df['CabinSide'].fillna(df.groupby('Group')['CabinSide'].transform('first'))
    df['CabinSide'] = df['CabinSide'].fillna('S')
    
    df['CabinDeck'] = df['CabinDeck'].fillna(df.groupby('Group')['CabinDeck'].transform('first'))
    df['CabinDeck'] = df['CabinDeck'].fillna('F')
    
    df['CabinNum'] = df.groupby(['CabinDeck', 'CabinSide'])['CabinNum'].transform(lambda x: x.fillna(x.median()))
    df['CabinNum'] = df['CabinNum'].fillna(df['CabinNum'].median())
    
    df['CabinRegion'] = pd.qcut(df['CabinNum'], q=7, labels=False).astype(str)
    
    # --- 5. Age & VIP ---
    df['Age'] = df.groupby('HomePlanet')['Age'].transform(lambda x: x.fillna(x.median()))
    df['AgeGroup'] = pd.cut(df['Age'], bins=[-1, 12, 18, 25, 40, 100], labels=['Child', 'Teen', 'Youth', 'Adult', 'Elder']).astype(str)
    
    df['VIP'] = df['VIP'].fillna(False).astype(int)
    df['Destination'] = df['Destination'].fillna('TRAPPIST-1e')
    
    # --- Log Transforms ---
    for col in expense_cols + ['TotalExpense']:
        df[col] = np.log1p(df[col])
        
    # --- Drop unnecessary ---
    df = df.drop(['PassengerId', 'Group', 'Cabin', 'Name', 'Surname'], axis=1)
    
    cat_cols = ['HomePlanet', 'Destination', 'CabinDeck', 'CabinSide', 'CabinRegion', 'AgeGroup']
    for col in cat_cols:
        df[col] = df[col].astype(str)
        
    X_train = df.iloc[:train_size]
    X_test = df.iloc[train_size:]
    
    return X_train, y, X_test, cat_cols

def objective(trial, X, y, cat_features):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 30.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 150),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
        'bootstrap_type': 'MVS',
        'loss_function': 'Logloss',
        'eval_metric': 'Accuracy',
        'od_type': 'Iter',
        'od_wait': 50,
        'verbose': False,
        'random_seed': 42
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_va, y_va, cat_features=cat_features)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100)

        preds = model.predict(val_pool)
        acc = accuracy_score(y_va, preds)
        fold_accuracies.append(acc)
        
        # Report back to Optuna for pruning
        trial.report(acc, len(fold_accuracies))
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(fold_accuracies)

if __name__ == "__main__":
    train_df = pd.read_csv('datasets/spaceship-titanic/train.csv')
    test_df = pd.read_csv('datasets/spaceship-titanic/test.csv')

    print("Preprocessing...")
    X_train, y_train, X_test, cat_cols = preprocess_data(train_df, test_df)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    print("Optuna tuning CatBoost (5000 trials + Pruning)...")
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(lambda trial: objective(trial, X_train, y_train, cat_cols), n_trials=5000)

    best_params = study.best_params
    best_params.update({
        'bootstrap_type': 'MVS',
        'loss_function': 'Logloss',
        'eval_metric': 'Accuracy',
        'od_type': 'Iter',
        'od_wait': 50,
        'verbose': False,
        'random_seed': 42
    })
    
    print(f"Best CV Acc: {study.best_value:.4f}")
    print("Training final model across 5 folds and ensembling predictions for robust test output...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_preds_probs = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"  Training Fold {fold+1}...")
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        val_pool = Pool(X_va, y_va, cat_features=cat_cols)
        
        model = CatBoostClassifier(**best_params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
        
        # Predict on actual test.csv
        test_pool = Pool(X_test, cat_features=cat_cols)
        test_preds_probs += model.predict_proba(test_pool)[:, 1] / 5.0
        
    final_preds = (test_preds_probs > 0.5).astype(bool)

    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': final_preds
    })
    submission.to_csv('submission_best.csv', index=False)
    print("Done! Submission saved to 'submission_best.csv'")