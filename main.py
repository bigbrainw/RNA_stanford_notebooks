import statistics
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

print("=" * 70)
print("VARIANCE CALCULATION FOR STUDENTS AGE STATS")
print("=" * 70)

ages = [18, 19, 20, 21, 22]
frequencies = [10, 21, 24, 8, 3]
total_students = sum(frequencies)

print(f"\nData:")
print(f"{'Age (xi)':<15} {'Frequency (ni)':<20}")
print("-" * 35)
for age, freq in zip(ages, frequencies):
    print(f"{age:<15} {freq:<20}")
print("-" * 35)
print(f"{'Total (N)':<15} {total_students:<20}")

age_data = []
for age, freq in zip(ages, frequencies):
    age_data.extend([age] * freq)

mean = statistics.mean(age_data)
print(f"\nMean (μ) = {mean:.4f}")

variance_freq = 0
for xi, ni in zip(ages, frequencies):
    variance_freq += (xi - mean) ** 2 * ni
variance_freq = variance_freq / total_students

variance_standard = statistics.variance(age_data) if len(age_data) > 1 else 0
variance_population = statistics.pvariance(age_data)

variance_manual = sum((x - mean) ** 2 for x in age_data) / len(age_data)

std_dev = np.sqrt(variance_freq)

print("\n" + "=" * 70)
print("VARIANCE CALCULATIONS")
print("=" * 70)

print(f"\nUsing Frequency Distribution Formula:")
print(f"  σ² = Σ(xi - μ)²ni / N")
print(f"\n  Step-by-step calculation:")
for xi, ni in zip(ages, frequencies):
    diff_squared = (xi - mean) ** 2
    contribution = diff_squared * ni
    print(f"    (xi={xi} - μ={mean:.4f})² × ni={ni} = {diff_squared:.4f} × {ni} = {contribution:.4f}")

print(f"\n  Sum = {sum((xi - mean) ** 2 * ni for xi, ni in zip(ages, frequencies)):.4f}")
print(f"  Variance (σ²) = {variance_freq:.4f}")

print(f"\nVerification using standard formula:")
print(f"  σ² = Σ(xi - μ)² / N = {variance_manual:.4f}")
print(f"  (Using statistics.pvariance): {variance_population:.4f}")

print(f"\nStandard Deviation (σ) = √σ² = {std_dev:.4f}")

print("\n" + "=" * 70)
print("DETAILED CALCULATION TABLE")
print("=" * 70)
print(f"{'Age (xi)':<10} {'Freq (ni)':<12} {'(xi - μ)':<15} {'(xi - μ)²':<15} {'(xi - μ)²ni':<15}")
print("-" * 70)
for xi, ni in zip(ages, frequencies):
    diff = xi - mean
    diff_squared = diff ** 2
    contribution = diff_squared * ni
    print(f"{xi:<10} {ni:<12} {diff:<15.4f} {diff_squared:<15.4f} {contribution:<15.4f}")
print("-" * 70)
total_contribution = sum((xi - mean) ** 2 * ni for xi, ni in zip(ages, frequencies))
print(f"{'Total':<10} {total_students:<12} {'':<15} {'':<15} {total_contribution:<15.4f}")
print(f"\nVariance = {total_contribution} / {total_students} = {variance_freq:.4f}")

plt.figure(figsize=(10, 6))
plt.bar(ages, frequencies, edgecolor='black', alpha=0.7, color='steelblue', width=0.8)
plt.xlabel('Student Age', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Histogram of Students Age Distribution', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean (μ) = {mean:.2f}')

plt.axvline(mean - std_dev, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'μ ± σ')
plt.axvline(mean + std_dev, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

plt.legend()
plt.xticks(ages)
plt.tight_layout()
plt.savefig('students_age_histogram.png', dpi=150, bbox_inches='tight')
print(f"\nHistogram saved as 'students_age_histogram.png'")
plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Statistic':<30} {'Value':<20}")
print("-" * 50)
print(f"{'Mean (μ)':<30} {mean:.4f}")
print(f"{'Variance (σ²)':<30} {variance_freq:.4f}")
print(f"{'Standard Deviation (σ)':<30} {std_dev:.4f}")
print(f"{'Total Students (N)':<30} {total_students}")
