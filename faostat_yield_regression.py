import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_FILE = "faostat_data.csv"
NORMALIZATION_METHOD = "zscore"

if DATA_FILE.endswith(".xlsx"):
    df = pd.read_excel(DATA_FILE)
else:
    df = pd.read_csv(DATA_FILE)

df = df[df["Element"].str.strip().str.lower() == "yield"]
df = df[["Year", "Value"]].rename(columns={"Value": "Yield"})
df = df.dropna()
df["Year"] = df["Year"].astype(int)
df["Yield"] = df["Yield"].astype(float)
df = df[(df["Year"] >= 1960) & (df["Year"] <= 2025)]
df = df.sort_values("Year").reset_index(drop=True)

print(df.head())
print(f"Total rows: {len(df)}")

if NORMALIZATION_METHOD == "zscore":
    scaler = StandardScaler()
else:
    scaler = MinMaxScaler()

df["Year_norm"] = scaler.fit_transform(df[["Year"]])
df["Yield_norm"] = scaler.fit_transform(df[["Yield"]])
print(df[["Year", "Year_norm", "Yield", "Yield_norm"]].head())

X = df[["Year"]].values
y = df["Yield"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)

weight = model.coef_[0]
bias = model.intercept_

print("\nPart A - Linear Regression")
print(f"yield = {weight:.4f} * year + ({bias:.4f})")

y_pred_test = model.predict(X_test)

future_years = np.array([[2027], [2028], [2029], [2030]])
future_predictions = model.predict(future_years)

print("\nPredictions 2027-2030")
for year, pred in zip(future_years.flatten(), future_predictions):
    print(f"{year}: {pred:.2f}")

mse = mean_squared_error(y_test, y_pred_test)
print(f"\nPart B - MSE = {mse:.4f}")

print(f"\nPart C - weight = {weight:.4f}, bias = {bias:.4f}")

rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("\nExtra metrics")
print(f"RMSE = {rmse:.4f}")
print(f"MAE = {mae:.4f}")
print(f"R2 = {r2:.4f}")

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

print("\nPolynomial regression (degree 2)")
print(f"MSE = {mean_squared_error(y_test, y_pred_poly):.4f}")
print(f"RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_poly)):.4f}")
print(f"MAE = {mean_absolute_error(y_test, y_pred_poly):.4f}")
print(f"R2 = {r2_score(y_test, y_pred_poly):.4f}")

future_years_poly = poly.transform(future_years)
future_poly_predictions = poly_model.predict(future_years_poly)
print("\nPolynomial predictions 2027-2030")
for year, pred in zip(future_years.flatten(), future_poly_predictions):
    print(f"{year}: {pred:.2f}")

df["Yield_MA5"] = df["Yield"].rolling(window=5, min_periods=1).mean()

X_multi = df[["Year", "Yield_MA5"]].values
y_multi = df["Yield"].values

Xm_train, Xm_test, ym_train, ym_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

multi_model = LinearRegression()
multi_model.fit(Xm_train, ym_train)
ym_pred = multi_model.predict(Xm_test)

print("\nMultivariate regression")
print(f"weights = {multi_model.coef_}")
print(f"bias = {multi_model.intercept_:.4f}")
print(f"MSE = {mean_squared_error(ym_test, ym_pred):.4f}")
print(f"RMSE = {np.sqrt(mean_squared_error(ym_test, ym_pred)):.4f}")
print(f"MAE = {mean_absolute_error(ym_test, ym_pred):.4f}")
print(f"R2 = {r2_score(ym_test, ym_pred):.4f}")