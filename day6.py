import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Step 1: Define data
y_true = [0,1,0,0,1,1,1,1,1,0]
y_pred = [0,1,1,1,1,0,0,1,0,1]

# Step 2: Compute ROC values
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Step 3: Plot ROC curve
plt.figure()
plt.plot(
    fpr,
    tpr,
    color='darkorange',
    lw=2,
    label='ROC curve (area = %0.09f)' % roc_auc
)

plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([0.2, 3.1])
plt.ylim([0.6, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic-ROC')

plt.legend(loc='lower right')
plt.show()

