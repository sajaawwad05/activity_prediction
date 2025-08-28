import pandas as pd
import numpy as np
import os


folder = r"C:\Users\test\OneDrive\Desktop\jupyter_projects"


orientation_file = os.path.join(folder, "OrientationStds.csv")
gravity_file = os.path.join(folder, "GravityStds.csv")
gyroscope_file = os.path.join(folder, "GyroscopeStds.csv")
linear_acc_file = os.path.join(folder, "Linear AccelerometerStds.csv")


orientation = pd.read_csv(orientation_file)
gravity = pd.read_csv(gravity_file)
gyroscope = pd.read_csv(gyroscope_file)
linear_acc = pd.read_csv(linear_acc_file)


orientation = orientation[['x', 'y', 'z']]
orientation = orientation.astype(float) * (np.pi)

# removing Time from the columns
for df_name, df in zip(['gravity','gyroscope','linear_acc'], [gravity, gyroscope, linear_acc]):
    if "Time" in df.columns or "time" in df.columns:
        df.drop(df.columns[0], axis=1, inplace=True)
    
    if df_name == 'gravity':
        
        df.iloc[:, -3:] = df.iloc[:, -3:] / 9.81
    df = df.iloc[:, -3:]
    if df_name == 'gravity':
        gravity = df
    elif df_name == 'gyroscope':
        gyroscope = df
    else:
        linear_acc = df


min_len = min(len(orientation), len(gravity), len(gyroscope), len(linear_acc))
orientation = orientation.iloc[:min_len]
gravity = gravity.iloc[:min_len]
gyroscope = gyroscope.iloc[:min_len]
linear_acc = linear_acc.iloc[:min_len]


merged = pd.concat([orientation, gravity, gyroscope, linear_acc], axis=1)
merged.columns = [
    "attitude.roll", "attitude.pitch", "attitude.yaw",
    "gravity.x", "gravity.y", "gravity.z",
    "rotationRate.x", "rotationRate.y", "rotationRate.z",
    "userAcceleration.x", "userAcceleration.y", "userAcceleration.z"
]





target_rows = 20
num_original_rows = len(merged)


averaged_data = pd.DataFrame()
for col in merged.columns:
    col_values = merged[col].values
    indices = np.linspace(0, num_original_rows, target_rows + 1, dtype=int)
    averaged_col = []
    for i in range(target_rows):
        start_idx = indices[i]
        end_idx = indices[i + 1]
        window_mean = col_values[start_idx:end_idx].mean()
        averaged_col.append(window_mean)
    averaged_data[col] = averaged_col


output_file = os.path.join(folder, "standing.csv")
averaged_data.to_csv(output_file, index=False)


