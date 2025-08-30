# BigMart Sales Prediction 
#Saptarshi Ray


import pandas as pd
import numpy as np

# Sklearn + ML libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 1. Load Data

train = pd.read_csv("train_v9rqX0R.csv")
test = pd.read_csv("test_AbJTz2l.csv")
sample = pd.read_csv("sample_submission_8RXa3c6.csv")


# 2. Handle Missing Values

# Item_Weight: fill missing with median within each Item_Type
train['Item_Weight'].fillna(train.groupby('Item_Type')['Item_Weight'].transform('median'), inplace=True)
test['Item_Weight'].fillna(test.groupby('Item_Type')['Item_Weight'].transform('median'), inplace=True)

# Outlet_Size: fill missing with most frequent (mode) value within Outlet_Type
train['Outlet_Size'].fillna(
    train.groupby('Outlet_Type')['Outlet_Size'].transform(lambda x: x.mode()[0] if not x.mode().empty else "Medium"),
    inplace=True
)
test['Outlet_Size'].fillna(
    test.groupby('Outlet_Type')['Outlet_Size'].transform(lambda x: x.mode()[0] if not x.mode().empty else "Medium"),
    inplace=True
)


# 3. Data Cleaning

# Item_Fat_Content has inconsistent labels, unify them
def clean_fat(x):
    if str(x).lower() in ['low fat', 'lf']:
        return 'Low Fat'
    elif str(x).lower() in ['reg']:
        return 'Regular'
    return x

train['Item_Fat_Content'] = train['Item_Fat_Content'].apply(clean_fat)
test['Item_Fat_Content'] = test['Item_Fat_Content'].apply(clean_fat)


# 4. Feature Engineering

# Outlet_Age: how old the outlet is as of 2013
train['Outlet_Age'] = 2013 - train['Outlet_Establishment_Year']
test['Outlet_Age'] = 2013 - test['Outlet_Establishment_Year']

# Item_Category: first 2 chars of Item_Identifier (e.g., FD, DR, NC)
train['Item_Category'] = train['Item_Identifier'].str[0:2]
test['Item_Category'] = test['Item_Identifier'].str[0:2]


# 5. Encode Categorical Variables

# Encode categorical columns with LabelEncoder
le = LabelEncoder()
cat_cols = [
    'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
    'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Category'
]

for col in cat_cols:
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Drop ID columns not useful for training
train = train.drop(columns=['Item_Identifier'])
test = test.drop(columns=['Item_Identifier'])


# 6. Prepare Features and Target

X = train.drop(columns=['Item_Outlet_Sales'])
y = train['Item_Outlet_Sales']

# Log-transform the target to stabilize variance
y_log = np.log1p(y)

# Scale the features for NN models (not needed for trees, but harmless)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)


# 7. Define Base Models

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

xgb = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lgbm = LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# Simple neural network builder
def build_nn(input_dim, activation="swish"):
    """
    Creates a small feed-forward neural network with dropout.
    Two activation variants are tested: Swish and ELU.
    """
    if activation == "leaky_relu":
        act_layer = layers.LeakyReLU(alpha=0.1)
    else:
        act_layer = activation

    model = keras.Sequential([
        layers.Dense(256, activation=act_layer, input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(128, activation=act_layer),
        layers.Dropout(0.2),
        layers.Dense(1, activation='linear')  # regression output
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return model

# Register base models
base_models = [
    ("RandomForest", rf),
    ("XGBoost", xgb),
    ("LightGBM", lgbm),
    ("NeuralNet_Swish", "NN_swish"),
    ("NeuralNet_ELU", "NN_elu")
]


# 8. Stacking with K-Fold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# oof_preds = out-of-fold predictions for training meta-model
oof_preds = np.zeros((len(X), len(base_models)))
# test_preds = averaged predictions for test set
test_preds = np.zeros((len(test), len(base_models)))

# Perform 5-fold stacking
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f" Fold {fold+1}")
    
    # Split train/val sets
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

    # Also provide scaled versions for NN models
    X_train_scaled, X_val_scaled = X_scaled[train_idx], X_scaled[val_idx]

    # Train each base model
    for i, (name, model) in enumerate(base_models):
        if model == "NN_swish":
            # Train NN with Swish activation
            nn = build_nn(X_train_scaled.shape[1], activation="swish")
            callbacks = [
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            nn.fit(X_train_scaled, y_train,
                   validation_data=(X_val_scaled, y_val),
                   epochs=150, batch_size=64,
                   callbacks=callbacks, verbose=0)
            # Store predictions (convert back with expm1)
            oof_preds[val_idx, i] = np.expm1(nn.predict(X_val_scaled).ravel())
            test_preds[:, i] += np.expm1(nn.predict(test_scaled).ravel()) / kf.n_splits

        elif model == "NN_elu":
            # Train NN with ELU activation
            nn = build_nn(X_train_scaled.shape[1], activation="elu")
            callbacks = [
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            nn.fit(X_train_scaled, y_train,
                   validation_data=(X_val_scaled, y_val),
                   epochs=150, batch_size=64,
                   callbacks=callbacks, verbose=0)
            oof_preds[val_idx, i] = np.expm1(nn.predict(X_val_scaled).ravel())
            test_preds[:, i] += np.expm1(nn.predict(test_scaled).ravel()) / kf.n_splits

        else:
            # Tree-based models (RF, XGB, LGBM)
            model.fit(X_train, y_train)
            oof_preds[val_idx, i] = np.expm1(model.predict(X_val))
            test_preds[:, i] += np.expm1(model.predict(test)) / kf.n_splits


# 9. Meta-Model (Ridge Regression)

meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_preds, y)

# Show Ridge weights for each base model
weights = dict(zip([name for name, _ in base_models], meta_model.coef_))
print("\n Ridge Meta-Model Weights:")
for model, w in weights.items():
    print(f"   {model}: {w:.4f}")


# 10. Final Predictions

# Ridge makes the final blended prediction
final_preds = meta_model.predict(test_preds).clip(0)

# Create submission file
submission = sample.copy()
submission['Item_Outlet_Sales'] = final_preds
submission.to_csv("bigmart_submission_stacking_dualNN.csv", index=False)

print("Submission created: bigmart_submission_stacking_dualNN.csv")
