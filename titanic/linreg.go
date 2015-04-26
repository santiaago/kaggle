package main

import (
	"fmt"

	"github.com/santiaago/kaggle/data"
	"github.com/santiaago/kaggle/itertools"
	"github.com/santiaago/ml"
	"github.com/santiaago/ml/linreg"
)

// linregTest sets the Survived field of each passenger in the passenger array
// with respect to the prediction set by the linear regression 'linreg' passed as argument.
func linregTest(model *modelContainer, dc data.Container) (predictions []int) {

	fd := dc.Filter(model.Features)

	for i := 0; i < len(fd); i++ {

		x := []float64{1}
		x = append(x, fd[i][:len(fd[i])-1]...)

		if model.Model.HasTransform {
			x = model.Model.TransformFunction(x)
		}

		gi, err := model.Model.Predict(x)
		if err != nil {
			predictions = append(predictions, 0)
			continue
		}

		if ml.Sign(gi) == float64(1) {
			predictions = append(predictions, 1)
		} else {
			predictions = append(predictions, 0)
		}
	}
	return
}

// linregVectorsOfInterval returns an array functions.
// These functions return an array of linear regression and the corresponding features used.
func linregAllCombinations() (funcs []func(data.Container) modelContainers) {
	funcs = []func(dc data.Container) modelContainers{
		func(dc data.Container) modelContainers {
			return linregCombinations(dc, 2)
		},
		func(dc data.Container) modelContainers {
			return linregCombinations(dc, 3)
		},
		func(dc data.Container) modelContainers {
			return linregCombinations(dc, 4)
		},
		func(dc data.Container) modelContainers {
			return linregCombinations(dc, 5)
		},
		func(dc data.Container) modelContainers {
			return linregCombinations(dc, 6)
		},
		func(dc data.Container) modelContainers {
			return linregCombinations(dc, 7)
		},
	}
	return
}

// linregCombinations creates a linear regression model for each combination of
// the feature vector with respect to the size param.
// It returns an array of linear regressions, one for each combination.
func linregCombinations(dc data.Container, size int) (models modelContainers) {

	combs := itertools.Combinations(dc.Features, size)

	for _, c := range combs {
		fd := dc.Filter(c)
		lr := linreg.NewLinearRegression()
		lr.InitializeFromData(fd)

		name := fmt.Sprintf("LinregModel-V-%d-%v", size, c)

		if err := lr.Learn(); err != nil {
			continue
		}

		models = append(models, NewModelContainer(lr, name, c))
	}
	return
}

func specificLinregFuncs() []func(dc data.Container) (*modelContainer, error) {
	return []func(dc data.Container) (*modelContainer, error){
		linregSexAge,
		linregPClassAge,
		linregPClassSex,
		linregSexAgePClass,
	}
}

func linregSexAgePClass(dc data.Container) (*modelContainer, error) {

	lr := linreg.NewLinearRegression()

	features := []int{passengerIndexSex, passengerIndexAge, passengerIndexPclass}
	fd := dc.Filter(features)
	lr.InitializeFromData(fd)
	name := "Sex Age PClass"

	if err := lr.Learn(); err != nil {
		return nil, err
	}
	return NewModelContainer(lr, name, features), nil
}

func linregSexAge(dc data.Container) (*modelContainer, error) {

	lr := linreg.NewLinearRegression()

	features := []int{passengerIndexSex, passengerIndexAge}
	fd := dc.Filter(features)
	lr.InitializeFromData(fd)
	name := "Sex Age"

	if err := lr.Learn(); err != nil {
		return nil, err
	}
	return NewModelContainer(lr, name, features), nil
}

func linregPClassAge(dc data.Container) (*modelContainer, error) {

	lr := linreg.NewLinearRegression()

	features := []int{passengerIndexAge, passengerIndexPclass}
	fd := dc.Filter(features)
	lr.InitializeFromData(fd)
	name := "PClass Age"

	if err := lr.Learn(); err != nil {
		return nil, err
	}
	return NewModelContainer(lr, name, features), nil
}

func linregPClassSex(dc data.Container) (*modelContainer, error) {

	lr := linreg.NewLinearRegression()

	features := []int{passengerIndexSex, passengerIndexPclass}
	fd := dc.Filter(features)
	lr.InitializeFromData(fd)
	name := "PClass Sex"
	if err := lr.Learn(); err != nil {
		return nil, err
	}
	return NewModelContainer(lr, name, features), nil
}
