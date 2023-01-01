from bert_score import score
def bert_score_sumary (original_text, summary_text):
    P, R, F1 = score([original_text], [summary_text], lang="vi", verbose=True)
    return P