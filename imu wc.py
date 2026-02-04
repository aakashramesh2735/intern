# CELL 1 — Import Required Libraries
import os
import pandas as pd

# CELL 2 — Define Subject Folder Path
subject_path = r"C:\Users\abish\Downloads\aakash main project\DUMMY IMU5\Subject01"

# CELL 3 — Detect Trial Excel Files (CORRECT)
trial_files = sorted([
    f for f in os.listdir(subject_path)
    if f.startswith("Trial") and f.endswith(".xlsx")
])

print(f"Total trials found: {len(trial_files)}")
for f in trial_files:
    print(f)

# CELL 4 — Initialize Data Container (MUST BE DICT)
xsens_raw_data = {}

# CELL 5 — Load ALL Trials + SAVE
output_root = r"C:\Users\abish\Downloads\aakash main project\DUMMY IMU"
os.makedirs(output_root, exist_ok=True)

for trial_file in trial_files:
    
    trial_name = os.path.splitext(trial_file)[0]
    file_path = os.path.join(subject_path, trial_file)
    
    print(f"\nLoading {trial_name}...")
    
    xls = pd.ExcelFile(file_path)
    
    joint_angles = pd.read_excel(xls, sheet_name="Joint Angles ZXY")
    gyro_data    = pd.read_excel(xls, sheet_name="Segment Angular Velocity")
    acc_data     = pd.read_excel(xls, sheet_name="Segment Acceleration")
    
    time = joint_angles["Time"] if "Time" in joint_angles.columns else joint_angles.index
    
    xsens_raw_data[trial_name] = {
        "time": time,
        "joint_angles": joint_angles,
        "gyro": gyro_data,
        "acc": acc_data
    }

    # ✅ CREATE TRIAL FOLDER (INSIDE LOOP)
    trial_out_dir = os.path.join(output_root, trial_name)
    os.makedirs(trial_out_dir, exist_ok=True)

    # ✅ SAVE FILES
    joint_angles.to_csv(os.path.join(trial_out_dir, "joint_angles.csv"), index=False)
    gyro_data.to_csv(os.path.join(trial_out_dir, "gyro.csv"), index=False)
    acc_data.to_csv(os.path.join(trial_out_dir, "acc.csv"), index=False)

    time.to_frame(name="Time").to_csv(
        os.path.join(trial_out_dir, "time.csv"),
        index=False
    )

    print(f"Saved data to {trial_out_dir}")



# CELL 6 — Verify One Trial
trial_check = "Trial01"

print(xsens_raw_data[trial_check].keys())
print("\nJoint Angles:")
print(xsens_raw_data[trial_check]["joint_angles"].head())

print("\nGyroscope:")
print(xsens_raw_data[trial_check]["gyro"].head())

print("\nAcceleration:")
print(xsens_raw_data[trial_check]["acc"].head())

# CELL 7 — Access Example
gyro_trial5 = xsens_raw_data["Trial05"]["gyro"]
time_trial5 = xsens_raw_data["Trial05"]["time"]
