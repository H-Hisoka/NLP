def calculate_cer(reference: str, hypothesis: str) -> float:
   """Character Error Rate"""
   if len(reference) == 0: return 0.0
   return editdistance.eval(reference, hypothesis) / len(reference)

def calculate_wer(reference: str, hypothesis: str) -> float:
   """Word Error Rate"""
   ref_words = reference.split()
   hyp_words = hypothesis.split()
   if len(ref_words) == 0: return 0.0
   return editdistance.eval(ref_words, hyp_words) / len(ref_words)

def evaluate_detection(correct_sentences, noisy_sentences, predicted_sentences):
   """Tính Precision, Recall, F1 cho bài toán phát hiện lỗi"""
   TP, FP, FN = 0, 0, 0
   for corr, noisy, pred in zip(correct_sentences, noisy_sentences, predicted_sentences):
       c_words = corr.split()
       n_words = noisy.split()
       p_words = pred.split()
       
       min_len = min(len(c_words), len(n_words), len(p_words))
       for i in range(min_len):
           is_actual_error = (n_words[i] != c_words[i])
           is_flagged_error = (n_words[i] != p_words[i])
           
           if is_actual_error and is_flagged_error:
               TP += 1
           elif not is_actual_error and is_flagged_error:
               FP += 1
           elif is_actual_error and not is_flagged_error:
               FN += 1
               
   precision = TP / (TP + FP + 1e-8)
   recall = TP / (TP + FN + 1e-8)
   f1 = 2 * precision * recall / (precision + recall + 1e-8)
   return precision, recall, f1