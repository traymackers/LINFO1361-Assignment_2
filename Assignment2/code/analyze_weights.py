from collections import defaultdict

def analyze_weights(file="weights.txt"):
    sums = defaultdict(float)
    counts = defaultdict(int)

    with open(file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] == "winner=0":  # on ne garde que les parties gagnées
                continue
            for part in parts[1:]:
                key, val = part.split("=")
                sums[key] += float(val)
                counts[key] += 1

    print("🔍 Moyenne des pondérations gagnantes :")
    for key in sums:
        avg = sums[key] / counts[key]
        print(f"{key} = {round(avg, 3)}")

if __name__ == "__main__":
    analyze_weights()
