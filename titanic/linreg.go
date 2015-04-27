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

func linregTest(model *ml.ModelContainer, dc data.Container) ([]float64, error) {

	fd := dc.Filter(model.Features)
	lr := model.Model.(*linreg.LinearRegression)
	return lr.Predictions(fd)
}

// linregVectorsOfInterval returns an array functions.
// These functions return an array of linear regression and the corresponding features used.
func linregAllCombinations() (funcs []func(data.Container) ml.ModelContainers) {
	funcs = []func(dc data.Container) ml.ModelContainers{
		func(dc data.Container) ml.ModelContainers {
			return linregCombinations(dc, 2)
		},
		func(dc data.Container) ml.ModelContainers {
			return linregCombinations(dc, 3)
		},
		func(dc data.Container) ml.ModelContainers {
			return linregCombinations(dc, 4)
		},
		func(dc data.Container) ml.ModelContainers {
			return linregCombinations(dc, 5)
		},
		func(dc data.Container) ml.ModelContainers {
			return linregCombinations(dc, 6)
		},
		func(dc data.Container) ml.ModelContainers {
			return linregCombinations(dc, 7)
		},
	}
	return
}

// linregCombinations creates a linear regression model for each combination of
// the feature vector with respect to the size param.
// It returns an array of linear regressions, one for each combination.
// todo(santiaago): move to ml

func linregCombinations(dc data.Container, size int) (models ml.ModelContainers) {

	combs := itertools.Combinations(dc.Features, size)

	for _, c := range combs {
		fd := dc.FilterWithPredict(c)
		lr := linreg.NewLinearRegression()
		lr.InitializeFromData(fd)

		name := fmt.Sprintf("LinregModel-V-%d-%v", size, c)

		if err := lr.Learn(); err != nil {
			continue
		}

		models = append(models, ml.NewModelContainer(lr, name, c))
	}
	return
}

func specificLinregFuncs() []func(dc data.Container) (*ml.ModelContainer, error) {
	return []func(dc data.Container) (*ml.ModelContainer, error){
		linregSexAge,
		linregPClassAge,
		linregPClassSex,
		linregSexAgePClass,
	}
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

func lookupModelWithRegularization(lr *linreg.LinearRegression) error {
	ein := lr.Ein()

	eAugs := []float64{}
	ks := []int{}
	for k := -50; k < 50; k++ {
		lr.K = k
		if err := lr.LearnWeightDecay(); err != nil {
			return err
		}
		eAugIn := lr.EAugIn()
		eAugs = append(eAugs, eAugIn)
		ks = append(ks, k)
		// fmt.Printf("EAugIn = %f for k = %d\n", eAugIn, lr.K)
	}

	i := argmin(eAugs)
	bestEAug := eAugs[i]

	if bestEAug < ein {
		fmt.Printf("found better Ein with regulirization.\n")
		fmt.Printf("Ein = %f\n", ein)
		fmt.Printf("EAugIn: %5.4f with k: %v\n", bestEAug, ks[i])

	}
	return nil
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
