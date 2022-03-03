import kfold_template
import pandas
from matplotlib import pyplot

from mpl_toolkits.mplot3d import Axes3D

import numpy

from sklearn import svm

dataset = pandas.read_csv("datasets/dataset_svm_3.csv")

dataset['x3'] = dataset['x1']**2 + dataset['x2']**2

target = dataset.iloc[:,0].values
data = dataset.iloc[:,1:4].values

pyplot.scatter(data[:,0], data[:,1], c=target)
pyplot.savefig("scatterplot_2.png")
pyplot.close()

x1 = data[:,0].reshape(-1,1)
x2 = data[:,1].reshape(-1,1)
x3 = data[:,2].reshape(-1,1)


fig = pyplot.figure()
fig1 = fig.add_subplot(111, projection="3d")
fig1.scatter(x1,x2,x3, c=target, depthshade = True)
# pyplot.savefig("scatterplot_3d.png")


machine = svm.SVC(kernel = "linear")
machine.fit(data, target)
coef = machine.coef_
intercept = machine.intercept_
print(coef)
print(intercept)



x1,x2 = numpy.meshgrid(x1,x2)
plane = -(coef[0][0]*x1 + coef[0][1]*x2 + intercept)/ coef[0][2]
fig_surface = fig.gca(projection="3d")
fig_surface.plot_surface(x1,x2,plane, alpha= 0.01)

pyplot.savefig("scatterplot_3d.png")





r2_scores, accuracy_scores, confusion_matrices = kfold_template.run_kfold(data, 
																		target, 
																		4, 
																		svm.SVC(kernel="linear"), 
																		1, 
																		1)
print(r2_scores)
print(accuracy_scores)
for i in confusion_matrices:
	print(i)








