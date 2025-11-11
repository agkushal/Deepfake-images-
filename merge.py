import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Load CSVs
vit  = pd.read_csv(r"C:\Users\Bebop\Desktop\deepfakeAI\Results\vit_test_prob.csv")
goog = pd.read_csv(r"C:\Users\Bebop\Desktop\deepfakeAI\Results\google_test_prob.csv")
res  = pd.read_csv(r"C:\Users\Bebop\Desktop\deepfakeAI\Results\inception_test_prob.csv")

# Normalize names for a clean join
for df in (vit, goog, res):
    df["Image_Name"] = df["Image_Name"].str.lower()

# Keep only needed cols & give each model unique prob column names
vit  = vit[["Image_Name","True_Label","Prob_fake","Prob_real"]].rename(
       columns={"Prob_fake":"Prob_fake_vit","Prob_real":"Prob_real_vit"})
goog = goog[["Image_Name","Prob_fake","Prob_real"]].rename(
       columns={"Prob_fake":"Prob_fake_goog","Prob_real":"Prob_real_goog"})
res  = res[["Image_Name","Prob_fake","Prob_real"]].rename(
       columns={"Prob_fake":"Prob_fake_res","Prob_real":"Prob_real_res"})

# Merge (left on ViT so True_Label comes from ViT only)
merged = vit.merge(goog, on="Image_Name").merge(res, on="Image_Name")

# Ground truth
y_true = merged["True_Label"].str.lower().map({"fake":1, "real":0})

# Ensemble (simple average of 3 fake probs)
merged["Ensemble_Prob_Fake"] = (
    merged["Prob_fake_vit"] + merged["Prob_fake_goog"] + merged["Prob_fake_res"]
) / 3.0
merged["Ensemble_Pred"] = (merged["Ensemble_Prob_Fake"] > 0.5).astype(int)

# Metrics
acc = accuracy_score(y_true, merged["Ensemble_Pred"])
f1  = f1_score(y_true, merged["Ensemble_Pred"])

print(f" Ensemble Accuracy : {acc:.4f}")
print(f" F1 Score: {f1:.4f}")
