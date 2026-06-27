import numpy as np
import matplotlib.pyplot as plt
def compute_clinical_metrics(y_true,y_pred_prob,threshold=0.5):
    y_pred=(y_pred_prob>=threshold).astype(int)
    TP=((y_pred==1)&(y_true==1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    TN = ((y_pred == 0) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    accuracy = (TP + TN) / len(y_true)

    return {
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'accuracy': accuracy
    }

def plot_metrics_at_thresholds(y_true,y_pred_prob):
    thresholds = np.arange(0.1, 1.0, 0.1)
    sensitivities = []
    ppvs = []

    for thresh in thresholds:
        metrics = compute_clinical_metrics(y_true, y_pred_prob, thresh)
        sensitivities.append(metrics['sensitivity'])
        ppvs.append(metrics['PPV'])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, sensitivities, 'b-o', label='Sensitivity')
    plt.plot(thresholds, ppvs, 'r-o', label='PPV')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Metric Value')
    plt.title('Sensitivity vs PPV Trade-off')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(sensitivities, ppvs, 'g-o')
    plt.xlabel('Sensitivity')
    plt.ylabel('PPV (Precision)')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    plt.savefig('outputs/day21_clinical_metrics.png')
    plt.show()

if __name__=="__main__":
    np.random.seed(42)
    y_true=np.zeros(500)
    y_true[:50]=1
    nodule_probs=np.random.beta(8,2,50)
    non_nodule_probs=np.random.beta(2,8,450)
    y_pred_prob = np.concatenate([nodule_probs, non_nodule_probs])
    metrics = compute_clinical_metrics(y_true, y_pred_prob)
    print("=" * 40)
    print("CLINICAL VALIDATION REPORT")
    print("=" * 40)
    print(f"Total patients:  500")
    print(f"Nodule cases:     50 (10%)")
    print(f"")
    print(f"TP (caught cancers):     {metrics['TP']}")
    print(f"FP (false alarms):       {metrics['FP']}")
    print(f"TN (correct negatives):  {metrics['TN']}")
    print(f"FN (missed cancers):     {metrics['FN']}")
    print(f"")
    print(f"Sensitivity:  {metrics['sensitivity']:.1%}")
    print(f"Specificity:  {metrics['specificity']:.1%}")
    print(f"PPV:          {metrics['PPV']:.1%}")
    print(f"NPV:          {metrics['NPV']:.1%}")
    print(f"Accuracy:     {metrics['accuracy']:.1%}")
    print(f"")
    print(f"Missed cancers (FN): {metrics['FN']} patients!")

    plot_metrics_at_thresholds(y_true, y_pred_prob)