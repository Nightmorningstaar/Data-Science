from sklearn.datasets import load_digits
dataset = load_digits()
X,Y = dataset.data,dataset.target

from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state = 0) # 70% training and 30% test

# FOR LINEAR SVM
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import confusion_matrix

clf = SVC(kernel = 'linear')
clf.fit(X_train,Y_train)
y_pred_svm = clf.predict(x_test)
confusion_matrics_svm = confusion_matrix(y_pred = y_pred_svm,y_true = y_test)
df_svm = pd.DataFrame(confusion_matrics_svm)

# FOR VISUALIZING THE WITH HEAT MAP IN ORDER TO HIGHLIGHT THE REALTIVE FREQ OF DIFFERENT TYPES OF ERROR(LINEAR SVM)
import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.metrics import accuracy_score

plt.figure(figsize = (5.5,4))
sns.heatmap(df_svm,annot = True)
plt.title('Svm Linear Kernal \nAccuracy :{0:.3f}'.format(accuracy_score(y_true = y_test,y_pred = y_pred_svm)))
plt.xlabel('Predicted label')
plt.ylabel('True label')

# FOR RBF SVM
clf2 = SVC(kernel = 'rbf')
clf2.fit(X_train,Y_train)
y_pred_svm2 = clf2.predict(x_test)
confusion_matrics_svm2 = confusion_matrix(y_pred = y_pred_svm,y_true = y_test)
df_svm2 = pd.DataFrame(confusion_matrics_svm2)

# FOR VISUALIZING THE WITH HEAT MAP IN ORDER TO HIGHLIGHT THE REALTIVE FREQ OF DIFFERENT TYPES OF ERROR(RBF SVM)

plt.figure(figsize = (5.5,4))
sns.heatmap(df_svm2,annot = True)
plt.title('Svm RBF Kernal \nAccuracy :{0:.3f}'.format(accuracy_score(y_true = y_test,y_pred = y_pred_svm2)))
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# MULTI CLASS CLASSIFIACTION REPORT
from sklearn.metrics import classification_report
print(classification_report(y_true = y_test,y_pred = y_pred_svm))
