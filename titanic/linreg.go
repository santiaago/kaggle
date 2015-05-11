package main

import (
	"fmt"

	"github.com/santiaago/kaggle/itertools"

	"github.com/santiaago/ml"
	"github.com/santiaago/ml/data"
	"github.com/santiaago/ml/linreg"
)

// linregTest sets the Survived field of each passenger in the passenger array
// with respect to the prediction set by the linear regression 'linreg' passed as argument.
//
func linregTest(model *ml.ModelContainer, dc data.Container) ([]float64, error) {

	fd := dc.Filter(model.Features)
	lr, ok := model.Model.(*linreg.LinearRegression)
	if !ok {
		return nil, fmt.Errorf("not a linear regression")
	}
	return lr.Predictions(fd)
}

// linregVectorsOfInterval returns an array functions.
// These functions return an array of linear regression and the corresponding features used.
//
func linregAllCombinations() (funcs []func(data.Container) ml.ModelContainers) {
	funcs = []func(dc data.Container) ml.ModelContainers{
		// func(dc data.Container) ml.ModelContainers {
		// 	return linregCombinations(dc, 2)
		// },
		func(dc data.Container) ml.ModelContainers {
			return linregCombinations(dc, 3)
		},
		func(dc data.Container) ml.ModelContainers {
			return linregCombinations(dc, 4)
		},
		func(dc data.Container) ml.ModelContainers {
			return linregCombinations(dc, 5)
		},
		// func(dc data.Container) ml.ModelContainers {
		// 	return linregCombinations(dc, 6)
		// },
		// func(dc data.Container) ml.ModelContainers {
		// 	return linregCombinations(dc, 7)
		// },
	}
	return
}

// linregCombinations creates a linear regression model for each combination of
// the feature vector with respect to the size param.
// It returns an array of linear regressions, one for each combination.
// todo(santiaago): move to ml
//
func linregCombinations(dc data.Container, size int) (models ml.ModelContainers) {

	combs := itertools.Combinations(dc.Features, size)

	for _, c := range combs {
		fd := dc.FilterWithPredict(c)
		lr := linreg.NewLinearRegression()
		lr.InitializeFromData(fd)

		name := fmt.Sprintf("linreg 1D %v", c)

		if err := lr.Learn(); err != nil {
			continue
		}

		models = append(models, ml.NewModelContainer(lr, name, c))
	}
	return
}

// linregWithRegularization returns a linear regression model if
// it is better than the model passed as argument, else it returns nil.
// todo(santiaago): move this to ml/linreg.
//
func linregWithRegularization(lr *linreg.LinearRegression) (*linreg.LinearRegression, error) {

	ein := lr.Ein()

	eAugs := []float64{}
	ks := []int{}

	// look for the best lambda = 10^-k
	for k := -50; k < 50; k++ {
		lr.K = k
		if err := lr.LearnWeightDecay(); err != nil {
			return nil, err
		}
		eAugIn := lr.EAugIn()
		eAugs = append(eAugs, eAugIn)
		ks = append(ks, k)
	}

	i := argmin(eAugs)
	bestEAug := eAugs[i]

	if bestEAug >= ein {
		return nil, nil
	}

	// better model found, make a copy of the model passed in.
	nlr := linreg.NewLinearRegression()
	*nlr = *lr
	nlr.K = ks[i]
	if err := nlr.LearnWeightDecay(); err != nil {
		return nil, err
	}

	// update Wn with WReg
	nlr.Wn = nlr.WReg

	return nlr, nil
}

func specificLinregFuncs() []func(dc data.Container) (*ml.ModelContainer, error) {
	return []func(dc data.Container) (*ml.ModelContainer, error){
		linregSexAge,
		linregPClassAge,
		linregPClassSex,
		linregSexAgePClass,
	}
}

func linregSexAge(dc data.Container) (*ml.ModelContainer, error) {

	lr := linreg.NewLinearRegression()

	features := []int{passengerIndexSex, passengerIndexAge}
	fd := dc.FilterWithPredict(features)
	lr.InitializeFromData(fd)
	name := "Sex Age"

	if err := lr.Learn(); err != nil {
		return nil, err
	}
	return ml.NewModelContainer(lr, name, features), nil
}

func linregPClassAge(dc data.Container) (*ml.ModelContainer, error) {

	lr := linreg.NewLinearRegression()

	features := []int{passengerIndexAge, passengerIndexPclass}
	fd := dc.FilterWithPredict(features)
	lr.InitializeFromData(fd)
	name := "PClass Age"

	if err := lr.Learn(); err != nil {
		return nil, err
	}
	return ml.NewModelContainer(lr, name, features), nil
}

func linregPClassSex(dc data.Container) (*ml.ModelContainer, error) {

	lr := linreg.NewLinearRegression()

	features := []int{passengerIndexSex, passengerIndexPclass}
	fd := dc.FilterWithPredict(features)
	lr.InitializeFromData(fd)
	name := "PClass Sex"
	if err := lr.Learn(); err != nil {
		return nil, err
	}
	return ml.NewModelContainer(lr, name, features), nil
}

func linregSexAgePClass(dc data.Container) (*ml.ModelContainer, error) {

	lr := linreg.NewLinearRegression()

	features := []int{passengerIndexSex, passengerIndexAge, passengerIndexPclass}
	fd := dc.FilterWithPredict(features)
	lr.InitializeFromData(fd)
	name := "Sex Age PClass"

	if err := lr.Learn(); err != nil {
		return nil, err
	}
	return ml.NewModelContainer(lr, name, features), nil
}
