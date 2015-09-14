## how to run titanic

~~~
GOPATH\src\github.com\santiaago\kaggle\titanic> .\titanic.exe -h
Usage of GOPATH\src\github.com\santiaago\kaggle\titanic\titanic.exe:
  -comb=0: number of features to try with all combinations.
  -dim=0: dimension of transformation.
  -e=false: defines if the program should export the used models defined in epath
  -epath="usedModels.json": json array with the description of the trained models.
  -i=false: defines if the program should import the models defined in ipath
  -ipath="models.json": path to a json array with models to use description.
  -linreg=false: train linear regressions.
  -logreg=false: train logistic regressions.
  -osvmK=false: override svmK.
  -osvmL=false: override svmL.
  -osvmT=false: override svmT.
  -rankEcv=false: writes a ranking.ecv.md file with the cross validation ranking of all processed models.
  -rankEin=false: writes a ranking.ein.md file with the in sample ranking of all processed models.
  -reg=false: train models with regularization.
  -specific=false: train specific models.
  -svm=false: train support vector machines.
  -svmK=1: number of block size that should be try for the svm pegasos algorithm.
  -svmKRange=1: range of number of block size that should be try for the svm pegasos algorithm. If k = 10, we will try a
ll values from 1 to k
  -svmL=0.001: lambda, regularization parameter.
  -svmT=1000: number of iterations for svm Pegasos algorithm.
  -temp="data/temp/": path of temp folder where all model results and rankings will be written.
  -test=false: run test on test source and write to predictions to files.
  -testSrc="data/test.csv": testing set.
  -top=10: exports the top N models
  -trainSrc="data/train.csv": training set.
  -trans=false: train models with transformations.
  -v=false: verbose: print additional output
~~~


