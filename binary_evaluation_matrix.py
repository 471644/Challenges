import sklearn.metrics as metrics
def perf_measure(y_actual, y_pred):
    eval_dict = {}
    cm = metrics.confusion_matrix(y_actual, y_pred)
    TN = cm[0][0]
    TP = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]
    prfs = metrics.precision_recall_fscore_support(y_actual, y_pred)
    eval_dict['Recall'] = [
        metrics.recall_score(y_actual, y_pred), prfs[1][0], prfs[1][1]
    ]
    eval_dict['Precision'] = [
        metrics.precision_score(y_actual, y_pred), prfs[0][0], prfs[0][1]
    ]
    eval_dict['False_Positive_Rate'] = [FP / prfs[3][0]]
    eval_dict['False_Negative_Rate'] = [FN / prfs[3][1]]
    eval_dict['F1_Score'] = [
        metrics.f1_score(y_actual, y_pred), prfs[2][0], prfs[2][1]
    ]
    eval_dict['Accuracy'] = [
        metrics.accuracy_score(y_actual, y_pred), TN / prfs[3][0],
        TP / prfs[3][1]
    ]
    eval_dict['Class_Count'] = [len(y_actual)/100, len([i for i in y_actual if i in [0, 'clean']])/100,
                                len([i for i in y_actual if i in [1,'malicious']])/100]
    eval_dict['Class_Ratio'] = [i/len(y_actual) for i in eval_dict['Class_Count']]
    return pd.DataFrame(eval_dict, index=['Overall', 0, 1]) * 100