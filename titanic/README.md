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


examples:

##### training and testing linear regression ranking by in sample error
~~~
> .\titanic.exe -test -linreg -specific -rankEin
EIn = 0.212121  linreg Sex Age PClass
EIn = 0.213244  linreg Sex Age
EIn = 0.213244  linreg PClass Sex
EIn = 0.298541  linreg PClass Age
~~~

##### training and testing linear regression ranking by cross validation error
~~~
> .\titanic.exe -test -linreg -specific -rankEcv
Leave 890 out of 891
Ecv = 0.212121  linreg Sex Age PClass
Ecv = 0.213244  linreg Sex Age
Ecv = 0.213244  linreg PClass Sex
Ecv = 0.298541  linreg PClass Age
~~~

##### training and testing linear regression with feature combination of size 6 rank by in sample error
~~~
> .\titanic.exe -test -linreg -comb=6 -rankEin
[6 7 8 9 10 11]/84
EIn = 0.203143  linreg 1D [2 4 5 6 7 9]
EIn = 0.206510  linreg 1D [2 4 6 7 9 11]
EIn = 0.208754  linreg 1D [2 4 5 6 7 11]
EIn = 0.208754  linreg 1D [2 4 5 6 9 11]
EIn = 0.210999  linreg 1D [4 5 6 7 9 11]
EIn = 0.212121  linreg 1D [2 4 5 7 9 11]
EIn = 0.289562  linreg 1D [2 5 6 7 9 11]
~~~

##### using `-trans` and `-dim`
Training and testing linear regression with feature combination of size **5** and
transforming the vectors with **5D** transformations rank by in sample error and getting the top **10** results.
**Note**:
If you get errors, it is usally because the feautre vector size does not match the size expected by the transformation size.
See [github.com/santiaago/ml/transform.go](github.com/santiaago/ml/transform.go) for more details.
~~~
> .\titanic.exe -test -linreg -comb=5 -trans -dim=5 -ran
kEin -top=10
[7 8 9 10 11]/126
size of transModel 160EIn = 0.193042    linreg 5D [2 4 6 7 11] transformed 2
EIn = 0.193042  linreg 5D [2 4 6 7 11] transformed 4
EIn = 0.198653  linreg 5D [2 4 6 7 11] transformed 6
EIn = 0.199776  linreg 5D [2 4 5 7 9] transformed 3
EIn = 0.200898  linreg 5D [2 4 5 6 11] transformed 4
EIn = 0.200898  linreg 5D [2 4 5 6 9] transformed 4
EIn = 0.200898  linreg 5D [2 4 5 6 9] transformed 2
EIn = 0.203143  linreg 5D [2 4 5 6 9] transformed 6
EIn = 0.203143  linreg 5D [2 4 5 6 9] transformed 1
EIn = 0.203143  linreg 5D [2 4 5 6 11] transformed 6
~~~
