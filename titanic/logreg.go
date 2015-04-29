package main

import (
	"fmt"

	"github.com/santiaago/ml"
	"github.com/santiaago/ml/data"
	"github.com/santiaago/ml/logreg"
)

// logregTest sets the Survived field of each passenger in the passenger array
// with respect to the prediction set by the linear regression 'linreg' passed as argument.
func logregTest(model *ml.ModelContainer, dc data.Container) ([]float64, error) {

	fd := dc.Filter(model.Features)
	lr, ok := model.Model.(*logreg.LogisticRegression)
	if !ok {
		return nil, fmt.Errorf("not a logistic regression")
	}
	return lr.Predictions(fd)
}

func specificLogregFuncs() []func(dc data.Container) (*ml.ModelContainer, error) {
	return []func(dc data.Container) (*ml.ModelContainer, error){
		logregSexAge,
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
