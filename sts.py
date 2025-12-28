import pandas as pd
import numpy as np

# Data dari Soal 2
X = np.array([4.7, 6.7, 9.5, 11.6, 13.5, 19.0, 19.8, 19.6])
Y = np.array([22.7, 32.0, 45.4, 54.0, 59.6, 73.7, 87.5, 96.1])

# Hitung rata-rata
X_mean = np.mean(X)
Y_mean = np.mean(Y)

# Deviasi dari rata-rata
x_dev = X - X_mean
y_dev = Y - Y_mean

# Kuadrat deviasi dan hasil kali deviasi
x2 = x_dev**2
y2 = y_dev**2
xy = x_dev * y_dev

# Buat tabel dalam DataFrame
df = pd.DataFrame({
    'X': X,
    'Y': Y,
    '(x)': x_dev,
    '(y)': y_dev,
    'x²': x2,
    'y²': y2,
    'xy': xy
})


sum_row = pd.DataFrame(df.sum()).T
sum_row.index = ['Jumlah']
df = pd.concat([df, sum_row])

print(df.round(4))
