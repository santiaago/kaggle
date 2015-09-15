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

##### using `-rankEin`
training and testing linear regression ranking by in sample error.
~~~
> .\titanic.exe -linreg -specific -rankEin
EIn = 0.212121  linreg Sex Age PClass
EIn = 0.213244  linreg Sex Age
EIn = 0.213244  linreg PClass Sex
EIn = 0.298541  linreg PClass Age
~~~

the `-rankEin` will create a `ranking.ein.md` in `\data\temp`

~~~
> ls .\data\temp
> .\titanic.exe -specific -linreg -rankEin
EIn = 0.212121  linreg Sex Age PClass
EIn = 0.213244  linreg Sex Age
EIn = 0.213244  linreg PClass Sex
EIn = 0.298541  linreg PClass Age
> ls .\data\temp
-a---        14/09/2015     16:35        203 ranking.ein.md
~~~

##### using `-rankEcv`
training and testing linear regression ranking by cross validation error.
~~~
> .\titanic.exe -linreg -specific -rankEcv
Leave 890 out of 891
Ecv = 0.212121  linreg Sex Age PClass
Ecv = 0.213244  linreg Sex Age
Ecv = 0.213244  linreg PClass Sex
Ecv = 0.298541  linreg PClass Age
~~~

##### using `-comb`
training and testing linear regression with feature combination of size 6 rank by in sample error
~~~
> .\titanic.exe -linreg -comb=6 -rankEin
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
> .\titanic.exe -linreg -comb=5 -trans -dim=5 -ran
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


#### using `-reg` flag
training linear regression on 6 feature combinations with and without regularization.
~~~
> .\titanic.exe -linreg -comb=6 -reg -rankEin -top=25
[6 7 8 9 10 11]/84
EIn = 0.203143  linreg 1D [2 4 5 6 7 9]
EIn = 0.206510  linreg 1D [2 4 6 7 9 11] regularized k 2
EIn = 0.206510  linreg 1D [2 4 6 7 9 11]
EIn = 0.208754  linreg 1D [2 4 5 6 7 11] regularized k 2
EIn = 0.208754  linreg 1D [2 4 5 6 7 11]
EIn = 0.208754  linreg 1D [2 4 5 6 9 11]
EIn = 0.208754  linreg 1D [2 4 5 6 9 11] regularized k 2
EIn = 0.210999  linreg 1D [4 5 6 7 9 11]
EIn = 0.210999  linreg 1D [4 5 6 7 9 11] regularized k 2
EIn = 0.212121  linreg 1D [2 4 5 7 9 11] regularized k 2
EIn = 0.212121  linreg 1D [2 4 5 7 9 11]
EIn = 0.289562  linreg 1D [2 5 6 7 9 11]
~~~

### using `-test` flag

`-test` flag creates the test files to be submited to **kaggle** in the format expected by kaggle.
See [here](https://www.kaggle.com/c/titanic/details/submission-instructions) for more details.

~~~
> ls .\data\temp
> .\titanic.exe -specific -linreg
> ls .\data\temp
> .\titanic.exe -specific -linreg -test
> ls .\data\temp
-a---        14/09/2015     16:36       2839 linreg PClass Age
-a---        14/09/2015     16:36       2839 linreg PClass Sex
-a---        14/09/2015     16:36       2839 linreg Sex Age
-a---        14/09/2015     16:36       2839 linreg Sex Age PClass
> tail '.\data\temp\linreg Sex Age'
1300,1
1301,1
1302,1
1303,1
1304,1
1305,0
1306,1
1307,0
1308,0
1309,0
~~~

#### using `-e` flag
Use this flag to export the models that you have trained.

~~~
> .\titanic.exe -linreg -comb=7 -e
> cat .\usedModels.json
[
    {
        "Model": 0,
        "TransformDimension": 0,
        "TransformID": 0,
        "Features": [
            2,
            4,
            5,
            6,
            7,
            9,
            11
        ],
        "Regularized": false,
        "K": 0,
        "T": 0,
        "L": 0
    }
]
~~~

#### using the `-epath` flag
If you wish to change the path use the `-epath` flag.
~~~
> .\titanic.exe -linreg -comb=7 -e -epath="example.json"
> cat .\example.json
[
    {
        "Model": 0,
        "TransformDimension": 0,
        "TransformID": 0,
        "Features": [
            2,
            4,
            5,
            6,
            7,
            9,
            11
        ],
        "Regularized": false,
        "K": 0,
        "T": 0,
        "L": 0
    }
]
~~~

#### using `-i` and `-ipath`

Similarly use the `-i` and `-ipath` to import models and test them or generate the test files.

~~~
> .\titanic.exe -linreg -comb=7 -i -ipath="example.json" -rankEin

EIn = 0.207632  linreg [2 4 5 6 7 9 11]
> .\titanic.exe -linreg -comb=7 -i -ipath="example.json" -test
> ls .\data\temp\
-a---        14/09/2015     16:58       2839 linreg [2 4 5 6 7 9 11]
-a---        14/09/2015     16:57         79 ranking.ein.md
~~~


#### use the verbose mode `-v` to see what is going on under the hood

~~~
> .\titanic.exe -linreg -comb=7 -rankEin -v
Starting training models
training linreg models

        training combinations
[5 6 7 8 9 10 11]/36

        Done, trained 1 combination models
Done. Trained 1 models
Start ranking models
Start ranking models by Ein
EIn = 0.207632  linreg 1D [2 4 5 6 7 9 11]
Done ranking models by Ein
Done ranking models
~~~

#### use logistic regression flag `-logreg`

~~~
> .\titanic.exe -logreg -comb=6 -rankEin
[6 7 8 9 10 11]/84
EIn = 0.196409  Logreg 1D [2 4 6 8 10 11] epochs-22
EIn = 0.199776  Logreg 1D [2 4 6 7 8 10] epochs-18
EIn = 0.199776  Logreg 1D [2 4 6 7 10 11] epochs-22
EIn = 0.199776  Logreg 1D [2 4 6 7 8 11] epochs-22
EIn = 0.200898  Logreg 1D [2 4 7 8 10 11] epochs-22
EIn = 0.206510  Logreg 1D [4 6 7 8 10 11] epochs-17
EIn = 0.208754  Logreg 1D [4 5 6 8 10 11] epochs-63
EIn = 0.208754  Logreg 1D [4 5 6 7 8 11] epochs-58
EIn = 0.208754  Logreg 1D [4 5 6 7 10 11] epochs-58
EIn = 0.209877  Logreg 1D [4 5 6 7 8 10] epochs-59
~~~

with regularization:
~~~
> .\titanic.exe -logreg -reg -comb=7 -rankEin
[5 6 7 8 9 10 11]/36
EIn = 0.187430  Logreg 1D [2 4 6 7 8 10 11] epochs-22 regularized k -2 epochs 1001
EIn = 0.198653  Logreg 1D [2 4 6 7 8 10 11] epochs-22 regularized k -3 epochs 1001
EIn = 0.199776  Logreg 1D [2 4 6 7 8 10 11] epochs-22 regularized k -4 epochs 22
EIn = 0.199776  Logreg 1D [2 4 6 7 8 10 11] epochs-22
EIn = 0.199776  Logreg 1D [2 4 6 7 8 10 11] epochs-22 regularized k -5 epochs 22
EIn = 0.205387  Logreg 1D [4 5 6 7 8 10 11] epochs-58 regularized k -4 epochs 1001
EIn = 0.208754  Logreg 1D [4 5 6 7 8 10 11] epochs-58 regularized k -5 epochs 1001
EIn = 0.208754  Logreg 1D [4 5 6 7 8 10 11] epochs-58
EIn = 0.212121  Logreg 1D [2 4 6 7 8 9 11] epochs-26
EIn = 0.212121  Logreg 1D [2 4 6 7 9 10 11] epochs-26 regularized k -5 epochs 28
~~~
