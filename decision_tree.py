# 1. Load Dataset dari CSV
def load_dataset(filepath):
    f = open(filepath, "r", encoding="utf-8")
    lines = f.readlines()
    f.close()

    headers = lines[0].strip().split(",")
    data = []

    for line in lines[1:]:
        values = line.strip().split(",")
        row = {}
        for i in range(len(headers)):
            col = headers[i]
            val = values[i]
            if col != "Keterangan":
                try:
                    row[col] = int(val)
                except:
                    row[col] = 0
            else:
                row[col] = val
        data.append(row)
    return data

# 2. Hitung Frekuensi Label
def count_labels(rows):
    counts = {}
    for row in rows:
        label = row["Keterangan"]
        counts[label] = counts.get(label, 0) + 1
    return counts

# 3. Log base 2
def log2(x):
    from math import log2
    return log2(x)

# 4. Hitung Entropy
def entropy(rows):
    total = len(rows)
    counts = count_labels(rows)
    result = 0.0
    for label in counts:
        p = counts[label] / total
        result -= p * log2(p)
    return result

# 5. Partisi Data
def partition(rows, feature, threshold):
    left, right = [], []
    for row in rows:
        if row[feature] <= threshold:
            left.append(row)
        else:
            right.append(row)
    return left, right

# 6. Information Gain
def info_gain(left, right, current_uncertainty):
    p = len(left) / (len(left) + len(right))
    return current_uncertainty - (p * entropy(left) + (1 - p) * entropy(right))

# 7. Cari Split Terbaik
def find_best_split(rows, features):
    best_gain = 0
    best_feature = None
    best_threshold = None
    current_uncertainty = entropy(rows)

    for feature in features:
        values = sorted(set([row[feature] for row in rows]))
        for val in values:
            left, right = partition(rows, feature, val)
            if len(left) == 0 or len(right) == 0:
                continue
            gain = info_gain(left, right, current_uncertainty)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = val
    return best_gain, best_feature, best_threshold

# 8. Node Tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction

# 9. Most Common Label
def most_common_label(labels):
    from collections import Counter
    return Counter(labels).most_common(1)[0][0]

# 10. Build Tree
def build_tree(rows, features, depth=0, max_depth=3):
    labels = [row["Keterangan"] for row in rows]
    if labels.count(labels[0]) == len(labels):
        return Node(prediction=labels[0])
    if depth >= max_depth:
        return Node(prediction=most_common_label(labels))

    gain, feature, threshold = find_best_split(rows, features)
    if gain == 0:
        return Node(prediction=most_common_label(labels))

    left, right = partition(rows, feature, threshold)
    left_branch = build_tree(left, features, depth + 1, max_depth)
    right_branch = build_tree(right, features, depth + 1, max_depth)

    return Node(feature, threshold, left_branch, right_branch)

# 11. Prediksi
def predict(row, node):
    if node.prediction is not None:
        return node.prediction
    if row[node.feature] <= node.threshold:
        return predict(row, node.left)
    else:
        return predict(row, node.right)

# 12. Confusion Matrix
def confusion_matrix(predictions, actuals):
    tp = fp = fn = tn = 0
    for pred, actual in zip(predictions, actuals):
        if actual == "Lolos":
            if pred == "Lolos":
                tp += 1
            else:
                fn += 1
        else:
            if pred == "Tidak Lolos":
                tn += 1
            else:
                fp += 1
    return tp, fp, fn, tn

# 13. Evaluasi & Cetak
def print_metrics(tp, fp, fn, tn):
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0

    print("\n📊 Confusion Matrix:")
    print(f"TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")
    print(f"Akurasi : {acc:.2f}")
    print(f"Presisi : {prec:.2f}")
    print(f"Recall  : {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

# 14. Evaluasi Verbose
def evaluate_verbose(tree, test_data):
    predictions = []
    actuals = []

    print("\n🧪 Contoh Prediksi:")
    for i, row in enumerate(test_data[:5]):
        pred = predict(row, tree)
        actual = row["Keterangan"]
        print(f"Data ke-{i+1}: Asli = {actual}, Prediksi = {pred}")
        predictions.append(pred)
        actuals.append(actual)

    for row in test_data[5:]:
        predictions.append(predict(row, tree))
        actuals.append(row["Keterangan"])

    tp, fp, fn, tn = confusion_matrix(predictions, actuals)
    print_metrics(tp, fp, fn, tn)

# 15. Cetak Tree
def print_tree(node, spacing=""):
    if node.prediction is not None:
        print(spacing + f"🟩 Predict: {node.prediction}")
        return
    print(spacing + f"🔀 {node.feature} <= {node.threshold}?")
    print(spacing + "├─ True:")
    print_tree(node.left, spacing + "│   ")
    print(spacing + "└─ False:")
    print_tree(node.right, spacing + "    ")

# 16. MAIN PROGRAM
def main():
    filename = "dataset.csv"
    data = load_dataset(filename)

    features = [
        "Ekonomi",
        "Geografi",
        "Kemampuan Bacaan dan Menulis",
        "Kemampuan Penalaran Umum",
        "Pengetahuan dan Pemahaman Umum",
        "Pengetahuan Kuantitatif",
        "Sejerah",
        "Sosiologi",
    ]

    train_data = data[:800]
    test_data = data[800:1000]

    print(f"Jumlah Data: {len(data)}")
    print("Fitur yang digunakan:", features, "\n")
    print("🔍 5 Contoh Data Pertama:")
    for d in data[:5]:
        print(d)

    tree = build_tree(train_data, features)
    print("\n🌳 Struktur Pohon Keputusan:")
    print_tree(tree)

    evaluate_verbose(tree, test_data)

# Run Program
if __name__ == "__main__":
    main()