"""
Analyze prediction differences between original (80 points) vs Exp1 (76.15%) vs Exp2 (71.95%)
"""
import pandas as pd
import numpy as np

# Load all submissions
print("Loading submissions...")
original = pd.read_csv("data/submission.csv")
exp1 = pd.read_csv("submission_exp1.csv")
exp2 = pd.read_csv("submission_exp2.csv")

# Get predicted classes
original_preds = original[["normal", "bacteria", "virus", "COVID-19"]].values.argmax(axis=1)
exp1_preds = exp1[["normal", "bacteria", "virus", "COVID-19"]].values.argmax(axis=1)
exp2_preds = exp2[["normal", "bacteria", "virus", "COVID-19"]].values.argmax(axis=1)

class_names = ["normal", "bacteria", "virus", "COVID-19"]

print("\n" + "="*80)
print("PREDICTION AGREEMENT ANALYSIS")
print("="*80)

# Agreement rates
orig_exp1_agree = (original_preds == exp1_preds).sum()
orig_exp2_agree = (original_preds == exp2_preds).sum()
exp1_exp2_agree = (exp1_preds == exp2_preds).sum()

total = len(original)

print(f"\nTotal test samples: {total}")
print(f"\nAgreement rates:")
print(f"  Original (80%) vs Exp1 (76.15%): {orig_exp1_agree}/{total} = {orig_exp1_agree/total*100:.2f}%")
print(f"  Original (80%) vs Exp2 (71.95%): {orig_exp2_agree}/{total} = {orig_exp2_agree/total*100:.2f}%")
print(f"  Exp1 (76.15%) vs Exp2 (71.95%):  {exp1_exp2_agree}/{total} = {exp1_exp2_agree/total*100:.2f}%")

# Class distribution
print("\n" + "="*80)
print("CLASS DISTRIBUTION")
print("="*80)

for i, name in enumerate(class_names):
    orig_count = (original_preds == i).sum()
    exp1_count = (exp1_preds == i).sum()
    exp2_count = (exp2_preds == i).sum()
    print(f"\n{name}:")
    print(f"  Original: {orig_count} ({orig_count/total*100:.1f}%)")
    print(f"  Exp1:     {exp1_count} ({exp1_count/total*100:.1f}%)")
    print(f"  Exp2:     {exp2_count} ({exp2_count/total*100:.1f}%)")

# Confusion between models
print("\n" + "="*80)
print("CONFUSION MATRIX: Original → Exp1")
print("="*80)

confusion_orig_exp1 = np.zeros((4, 4), dtype=int)
for i in range(total):
    confusion_orig_exp1[original_preds[i], exp1_preds[i]] += 1

print(f"\n{'':12s} {'normal':>10s} {'bacteria':>10s} {'virus':>10s} {'COVID-19':>10s}")
print("-" * 60)
for i, name in enumerate(class_names):
    print(f"{name:12s}", end="")
    for j in range(4):
        print(f"{confusion_orig_exp1[i,j]:>10d}", end="")
    print()

print("\n" + "="*80)
print("CONFUSION MATRIX: Original → Exp2")
print("="*80)

confusion_orig_exp2 = np.zeros((4, 4), dtype=int)
for i in range(total):
    confusion_orig_exp2[original_preds[i], exp2_preds[i]] += 1

print(f"\n{'':12s} {'normal':>10s} {'bacteria':>10s} {'virus':>10s} {'COVID-19':>10s}")
print("-" * 60)
for i, name in enumerate(class_names):
    print(f"{name:12s}", end="")
    for j in range(4):
        print(f"{confusion_orig_exp2[i,j]:>10d}", end="")
    print()

# Where do Exp1 and Exp2 differ from Original?
print("\n" + "="*80)
print("DISAGREEMENT ANALYSIS")
print("="*80)

disagreements_exp1 = (original_preds != exp1_preds)
disagreements_exp2 = (original_preds != exp2_preds)

print(f"\nExp1 disagrees with Original on {disagreements_exp1.sum()} samples")
print(f"Exp2 disagrees with Original on {disagreements_exp2.sum()} samples")

# Where both disagree
both_disagree = disagreements_exp1 & disagreements_exp2
both_agree_wrong = (exp1_preds == exp2_preds) & (original_preds != exp1_preds)

print(f"Both Exp1 & Exp2 disagree with Original: {both_disagree.sum()} samples")
print(f"Exp1 & Exp2 agree but differ from Original: {both_agree_wrong.sum()} samples")

# Show some examples where both new models agree but differ from original
print("\n" + "="*80)
print("EXAMPLES: Where Exp1 & Exp2 agree but differ from Original (80 pts baseline)")
print("="*80)

examples = []
for i in range(total):
    if both_agree_wrong[i]:
        examples.append({
            'filename': original.iloc[i]['new_filename'],
            'original': class_names[original_preds[i]],
            'exp1_exp2': class_names[exp1_preds[i]]
        })

print(f"\nShowing first 30 examples out of {len(examples)} total:")
for i, ex in enumerate(examples[:30]):
    print(f"{i+1:3d}. {ex['filename']:15s}  Original: {ex['original']:10s} → Exp1&Exp2: {ex['exp1_exp2']:10s}")

print("\n" + "="*80)
print("KEY FINDING")
print("="*80)

print(f"""
The new experiments (Exp1: 76.15%, Exp2: 71.95%) are performing WORSE than the
original baseline (80%). This suggests that either:

1. The original submission used a better model/configuration
2. There was overfitting to validation set (new models generalize differently)
3. Data preprocessing or augmentation differs from original approach

RECOMMENDATION:
Use the original submission (data/submission.csv) as the baseline, since it
achieved 80% which is BETTER than both Exp1 and Exp2.
""")
