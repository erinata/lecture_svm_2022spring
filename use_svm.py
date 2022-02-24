import kfold_template

import pandas

from matplotlib import pyplot

dataset = pandas.read_csv("datasets/dataset_svm_1.csv")

print(dataset)

target = dataset.iloc[:,0].values
data = dataset.iloc[:,1:3].values

# print(target)
# print(data)


pyplot.scatter(data[:,0], data[:,1], c=target)
pyplot.savefig("scatterplot_1.png")
pyplot.close()


