from sklearn import svm
from sklearn import datasets
# explore the dataset
iris = datasets.load_iris() #so from dataset i use loadiris func i import my iris dataset
print(type(iris)) #the op is bunch this bunch contain all the iris data and attributes
print(iris.data)#all features or all specification of your iris dataset
print(iris.feature_names)# this will print all the features names of your iris dataset
print(iris.target)# target is what is we are going to predict
print(iris.target_names)# this will print the target name as well

x = iris.data[:,2]#this is the independent vrbl in which we have to select first two features or 2 colums
y = iris.target# this the dependent vrbl my data points stored in dot target that is attribute of the
#dataset. It is the value i predict or which species my flower belongs to
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 ,random_state = 4)#random_state is
#each time when you run this code you give the same sampling so you can give any no
#creating the model so that when we give our new dataset it will clarify for which it belogs to

#we have to convert all arrays in 1-d
x_train_mod = x_train.reshape(-1,1)
x_test_mod  = x_test.reshape(-1,1)
y_train_mod  = y_test.reshape(-1,1)
y_test_mod  = y_test.reshape(-1,1)

model = svm.SVC(kernel = 'linear')#the kernel will convert your unseperable
#problem into seperable problem
model.fit(x_train_mod,y_train_mod)
y_pred_mod = model.predict(x_test_mod)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test_mod,y_pred_mod))