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

// linregCombinations creates a linear regression model for each combination of
// the feature vector with respect to the size param.
// It returns an array of linear regressions, one for each combination.
// todo(santiaago): move to ml
//
func linregCombinations(dc data.Container, size int) (models ml.ModelContainers) {

	combs := itertools.Combinations(dc.Features, size)

	for _, c := range combs {
		fmt.Printf("\r%v/%v", c, len(combs))
		fd := dc.FilterWithPredict(c)
		lr := linreg.NewLinearRegression()
		lr.InitializeFromData(fd)

		name := fmt.Sprintf("linreg 1D %v", c)

		if err := lr.Learn(); err == nil {
			models = append(models, ml.NewModelContainer(lr, name, c))
		}
	}
	fmt.Println()
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
	nlr.IsRegularized = true
	nlr.K = ks[i]
	if err := nlr.LearnWeightDecay(); err != nil {
		return nil, err
	}

	// update Wn with WReg
	nlr.Wn = nlr.WReg

	return nlr, nil
}

func specificLinregModels(dc data.Container) (models ml.ModelContainers) {

	cases := []struct {
		features []int
		name     string
	}{
		{
			[]int{passengerIndexSex, passengerIndexAge},
			"linreg Sex Age",
		},
		{
			[]int{passengerIndexAge, passengerIndexPclass},
			"linreg PClass Age",
		},

		{
			[]int{passengerIndexSex, passengerIndexPclass},
			"linreg PClass Sex",
		},
		{
			[]int{passengerIndexSex, passengerIndexAge, passengerIndexPclass},
			"linreg Sex Age PClass",
		},
	}

	for _, c := range cases {
		lr := linreg.NewLinearRegression()
		fd := dc.FilterWithPredict(c.features)
		lr.InitializeFromData(fd)

		if err := lr.Learn(); err != nil {
			continue
		}
		mc := ml.NewModelContainer(lr, c.name, c.features)
		models = append(models, mc)
	}
	return
}
