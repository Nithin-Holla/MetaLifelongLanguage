
def batch_encode(batch):
    text, labels = [], []
    for txt, lbl in batch:
        text.append(txt)
        labels.append(lbl)
    return text, labels
