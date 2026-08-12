def calculate_accuracy(true_positive, false_positive, true_negative, false_negative):
    """
    Calculate accuracy of a classification model.
    Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    total = true_positive + false_positive + true_negative + false_negative
    if total == 0:
        raise ValueError("Total samples must be greater than zero.")
    return (true_positive + true_negative) / total