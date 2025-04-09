from collections import defaultdict
import sys
sys.stdout.reconfigure(encoding="utf-8")

def analyze_weights(file="weights.txt"):
    from collections import defaultdict

    sums = defaultdict(float)
    counts = defaultdict(int)

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] == "winner=0":
                continue
            for part in parts[1:]:
                key, val = part.split("=")
                sums[key] += float(val)
                counts[key] += 1

    print("📊 Moyennes des poids gagnants :")
    phases = ["early", "mid", "late"]
    metrics = ["pieces", "mobility", "king_safety", "king_threat", "captures"]
    for phase in phases:
        print(f"\n🔹 {phase.capitalize()}:")
        for metric in metrics:
            k = f"{phase}_{metric}"
            if counts[k]:
                avg = sums[k] / counts[k]
                print(f"  {metric}: {round(avg, 3)}")

if __name__ == "__main__":
    analyze_weights()
