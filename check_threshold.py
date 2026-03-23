import sys

THRESHOLD = 0.6

with open("model_info.txt") as f:
    lines = dict(line.strip().split("=") for line in f if "=" in line)

run_id      = lines.get("run_id", "unknown")
val_accuracy = float(lines.get("val_accuracy", 0))

print(f"Run ID      : {run_id}")
print(f"val_accuracy: {val_accuracy:.4f}")
print(f"Threshold   : {THRESHOLD}")

if val_accuracy < THRESHOLD:
    print(f"FAILED: accuracy {val_accuracy:.4f} is below threshold {THRESHOLD}")
    sys.exit(1)

print(f"PASSED: accuracy {val_accuracy:.4f} meets threshold {THRESHOLD}")