from correction.error_detection import levenshtein

def calculate_top1_accuracy(correct_sentences, predicted_sentences):
    """Tính Top-1 Accuracy cho bài toán sửa lỗi"""
    correct_count = 0
    total_count = 0
    for corr, pred in zip(correct_sentences, predicted_sentences):
         c_words = corr.split()
         p_words = pred.split()
         
         min_len = min(len(c_words), len(p_words))
         for i in range(min_len):
              if c_words[i] == p_words[i]:
                correct_count += 1
              total_count += 1
              
    accuracy = correct_count / (total_count + 1e-8)
    return accuracy
def calculate_top3_accuracy(correct_sentences, predicted_sentences):
    """Tính Top-3 Accuracy cho bài toán sửa lỗi"""
    correct_count = 0
    total_count = 0
    for corr, pred in zip(correct_sentences, predicted_sentences):
         c_words = corr.split()
         p_words = pred.split()
         
         min_len = min(len(c_words), len(p_words))
         for i in range(min_len):
              if c_words[i] in p_words[i].split('|'):
                correct_count += 1
              total_count += 1
              
    accuracy = correct_count / (total_count + 1e-8)
    return accuracy

def calculate_precision(TP, FP):
    """Tính Precision"""
    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP + 1e-8)

def calculate_f1(precision, recall):
    """Tính F1 Score"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall + 1e-8)

def calculate_cer(prediction, ground_truth):
    """Tính Character Error Rate (CER)"""
    if len(ground_truth) == 0:
        return 0.0
    total_distance= 0
    total_characters = 0
    for pred, gt in zip(prediction, ground_truth):
        total_distance += levenshtein(pred, gt)
        total_characters += len(gt)
    return total_distance / total_characters if total_characters > 0 else 0.0

def calculate_exact_match(predictions, ground_truths):
    correct=0
    for pred, gt in zip(predictions, ground_truths):
        if pred == gt:
            correct += 1
    return correct / len(ground_truths) if len(ground_truths) > 0 else 0.0

def calculate_recall(TP, FN):
    """Tính Recall"""
    if TP + FN == 0:
        return 0.0
    return TP / (TP + FN + 1e-8)


def calculate_wer(predictions, ground_truths):
    total_distance = 0
    total_words = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_words = pred.split()
        gt_words = gt.split()
        rows = len(gt_words) + 1
        cols = len(pred_words) + 1
        dp = [[0] * cols for _ in range(rows)]
        for i in range(rows):
            dp[i][0] = i
        for j in range(cols):
            dp[0][j] = j
        for i in range(1, rows):
            for j in range(1, cols):
                if gt_words[i - 1] == pred_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1,
                        dp[i - 1][j - 1] + 1
                    )
        total_distance += dp[-1][-1]
        total_words += len(gt_words)
    return total_distance / total_words if total_words > 0 else 0.0


def evaluate_system(
    predictions,
    ground_truths,
    top3_predictions,
    TP,
    FP,
    FN
):
    """Đánh giá toàn bộ hệ thống"""
    top1_accuracy = calculate_top1_accuracy(
        ground_truths,
        predictions
    )
    top3_accuracy = calculate_top3_accuracy(
        ground_truths,
        top3_predictions
    )
    precision = calculate_precision(TP, FP)
    recall = calculate_recall(TP, FN)
    f1 = calculate_f1(
        precision,
        recall
    )
    cer = calculate_cer(
        predictions,
        ground_truths
    )
    wer = calculate_wer(
        predictions,
        ground_truths
    )
    exact_match = calculate_exact_match(
        predictions,
        ground_truths
    )
    return {
        "top1_accuracy": top1_accuracy,
        "top3_accuracy": top3_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cer": cer,
        "wer": wer,
        "exact_match": exact_match
    }    

