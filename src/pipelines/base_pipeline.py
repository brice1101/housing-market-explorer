from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def build_preprocessor(X):
    """Builds the column transformer for preprocessing."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, X.columns.drop("Year", errors='ignore'))])
    
    return preprocessor
