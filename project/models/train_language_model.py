from collections import Counter

class NGramLanguageModel:
   def __init__(self):
       self.unigram_counts = Counter()
       self.bigram_counts = Counter()
       self.vocab = set()
       
   def train(self, corpus):
       for sentence in corpus:
           words = ["<s>"] + sentence.lower().split() + ["</s>"]
           self.vocab.update(words)
           for w in words:
               self.unigram_counts[w] += 1
           for i in range(len(words) - 1):
               self.bigram_counts[(words[i], words[i+1])] += 1
               
   def get_bigram_prob(self, w1, w2, alpha=0.001):
       """Tính xác suất P(w2 | w1) có Laplace smoothing"""
       count_w1_w2 = self.bigram_counts.get((w1, w2), 0)
       count_w1 = self.unigram_counts.get(w1, 0)
       return (count_w1_w2 + alpha) / (count_w1 + alpha * len(self.vocab))