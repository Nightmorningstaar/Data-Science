# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use("ggplot")
#
# x = [1, 5, 1.5, 8, 1, 9]
# y = [2, 8, 1.8, 8, 0.6, 11]
# plt.scatter(x,y)
# plt.show()
#
# X = np.array([[1,2],
#              [5,8],
#              [1.5,1.8],
#              [8,8],
#              [1,0.6],
#              [9,11]])
# Y = [0,1,0,1,0,1]
#
# from sklearn.model_selection import train_test_split
# X_train,Y_train,x_test,y_test=train_test_split(X,Y,random_state = 0)
#
# from sklearn.svm import SVC
#
# liner_svc = SVC(kernel = 'linear',C = 1.0)
# liner_svc.fit(X,Y)
# # print('score : '.format(liner_svc.score(X,Y)))
#
# w = liner_svc.coef_[0]
# print(w)
#
# a = -w[0] / w[1]
#
# xx = np.linspace(0,12)
# yy = a * xx - liner_svc.intercept_[0] / w[1]
#
# h0 = plt.plot(xx, yy, 'k-', label = "non weighted div")
#
# plt.scatter(X[:, 0], X[:, 1], c = y)
# plt.legend()
# plt.show()
#
# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
#
# X = cancer.data
# Y = cancer.target

# Import train_test_split function
# from sklearn.model_selection import train_test_split
#
# # Split dataset into training set and test set
# X_train, x_test, Y_train, y_test = train_test_split(X,Y,
#         test_size=0.3,random_state = 1) # 70% training and 30% test

#Import svm model
# from sklearn.svm import SVC
#
# liner_svc = SVC(kernel='linear',C = 1.0) # Linear Kernel
# liner_svc.fit(X_train, Y_train)
# print("Accuracy: {:.3f}".format(liner_svc.score(x_test,y_test)))

# from sklearn import metrics
# y_pred = liner_svc.predict(x_test)
# print("Accuracy:",metrics.accuracy_score(y_test,y_pred))

from sklearn.datasets import make_blobs

X_D2,Y_D2 = make_blobs(n_samples = 100,n_features = 2,centers = 8
                       ,cluster_std = 1.1,random_state = 4)

Y_D2 = Y_D2 % 2

import matplotlib.pyplot as plt

plt.figure()
plt.title('Sample binary classifiaction problem with non linearly seperable classes')
plt.scatter(X_D2[:,0],X_D2[:,1],c = Y_D2,marker = 'o',s = 50)
# plt.show()

from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(X_D2,Y_D2,
        test_size=0.3,random_state = 0) # 70% training and 30% test


from python_prac.adspy_shared_utilities import plot_class_regions_for_classifier_subplot

fig,subaxes = plt.subplots(1,3,figsize = (18,6))

from sklearn.svm import SVC

for this_gamma,sublot in zip([0.01,1.0,10.0],subaxes):
    clf = SVC(kernel = 'rbf',gamma = this_gamma).fit(X_train,Y_train)
    title = 'Support Vector clf \n\
    :rbf kernel gamma = {:.3f}'.format(this_gamma)
    print('gamma :',this_gamma,'score :{:.3f}'.format(clf.score(x_test,y_test)))
    plot_class_regions_for_classifier_subplot(clf,X_train,Y_train,None,None,title,sublot)
    plt.tight_layout()

for this_gamma,this_axes in zip([0.01,1.0,10.0],subaxes):
        for this_C,subplot in zip([0.1,1,15,200],this_axes):
                clf = SVC(kernel = 'rbf',gamma = this_gamma,C = this_C).fit(X_train,Y_train)
                title = 'Support Vector clf \n\
                :rbf kernel gamma = {:.3f}'.format(this_gamma),'C :{:.3f}'.format(this_C)
                print('gamma :',this_gamma,'score :{:.3f}'.format(clf.score(x_test,y_test)))
                plot_class_regions_for_classifier_subplot(clf,X_train,Y_train,x_test,y_test,title,sublot)
                plt.tight_layout()
# plt.show()

from python_prac.adspy_shared_utilities import plot_class_regions_for_classifier

clf1 = SVC(kernel = 'poly',degree = 3)
clf2 = SVC(kernel = 'rbf',gamma = 1.0).fit(X_train,Y_train)
clf1.fit(X_train,Y_train)
# plot_class_regions_for_classifier(clf1,X_D2,Y_D2)
plot_class_regions_for_classifier(clf1,X_train,Y_train,None,None,'Support Vector Classifier:\nPoly kernel')
plot_class_regions_for_classifier(clf2,X_train,Y_train,None,None,'Support Vector Classifier:\nrbf kernel')

plt.show()
print('poly',clf1.score(x_test,y_test))
# print(clf2.score(x_test,y_test))