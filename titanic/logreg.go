package main

import (
	"fmt"

	"github.com/santiaago/kaggle/itertools"
	"github.com/santiaago/ml"
	"github.com/santiaago/ml/data"
	"github.com/santiaago/ml/logreg"
)

// logregTest sets the Survived field of each passenger in the passenger array
// with respect to the prediction set by the linear regression 'linreg' passed as argument.
//
func logregTest(model *ml.ModelContainer, dc data.Container) ([]float64, error) {

	fd := dc.Filter(model.Features)
	lr, ok := model.Model.(*logreg.LogisticRegression)
	if !ok {
		return nil, fmt.Errorf("not a logistic regression")
	}
	return lr.Predictions(fd)
}

// logregCombinations creates a logistic regression model for each combination of
// the feature vector with respect to the size param.
// It returns an array of linear regressions, one for each combination.
// todo(santiaago): move to ml
//
func logregCombinations(dc data.Container, size int) (models ml.ModelContainers) {

	combs := itertools.Combinations(dc.Features, size)

	for _, c := range combs {
		fmt.Printf("\r%v/%v", c, len(combs))
		fd := dc.FilterWithPredict(c)
		lr := logreg.NewLogisticRegression()
		lr.InitializeFromData(fd)

		if err := lr.Learn(); err != nil {
			continue
		}
		name := fmt.Sprintf("Logreg 1D %v epochs-%v", c, lr.Epochs)

		models = append(models, ml.NewModelContainer(lr, name, c))
	}
	fmt.Println()
	return
}

func specificLogregFuncs() []func(dc data.Container) (*ml.ModelContainer, error) {
	return []func(dc data.Container) (*ml.ModelContainer, error){
		logregSexAge,
		logregPClassAge,
		logregPClassSex,
		logregSexAgePClass,
	}
}

func logregSexAge(dc data.Container) (*ml.ModelContainer, error) {

	lr := logreg.NewLogisticRegression()

	features := []int{passengerIndexSex, passengerIndexAge}
	fd := dc.FilterWithPredict(features)
	lr.InitializeFromData(fd)
	name := "Sex Age logistic regression"

	if err := lr.Learn(); err != nil {
		return nil, err
	}
	return ml.NewModelContainer(lr, name, features), nil
}

func logregPClassAge(dc data.Container) (*ml.ModelContainer, error) {

	lr := logreg.NewLogisticRegression()

	features := []int{passengerIndexAge, passengerIndexPclass}
	fd := dc.FilterWithPredict(features)
	lr.InitializeFromData(fd)
	name := "PClass Age regression"

	if err := lr.Learn(); err != nil {
		return nil, err
	}
	return ml.NewModelContainer(lr, name, features), nil
}

func logregPClassSex(dc data.Container) (*ml.ModelContainer, error) {

	lr := logreg.NewLogisticRegression()

	features := []int{passengerIndexSex, passengerIndexPclass}
	fd := dc.FilterWithPredict(features)
	lr.InitializeFromData(fd)
	name := "PClass Sex regression"
	if err := lr.Learn(); err != nil {
		return nil, err
	}
	return ml.NewModelContainer(lr, name, features), nil
}

func logregSexAgePClass(dc data.Container) (*ml.ModelContainer, error) {

	lr := logreg.NewLogisticRegression()

	features := []int{passengerIndexSex, passengerIndexAge, passengerIndexPclass}
	fd := dc.FilterWithPredict(features)
	lr.InitializeFromData(fd)
	name := "Sex Age PClass regression"

	if err := lr.Learn(); err != nil {
		return nil, err
	}
	return ml.NewModelContainer(lr, name, features), nil
}
