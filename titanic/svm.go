package main

import (
	"fmt"

	"github.com/santiaago/kaggle/itertools"

	"github.com/santiaago/ml"
	"github.com/santiaago/ml/data"
	"github.com/santiaago/ml/svm"
)

// svmTest sets the Survived field of each passenger in the passenger array
// with respect to the prediction set by the svm 'svm' passed as argument.
//
func svmTest(model *ml.ModelContainer, dc data.Container) ([]float64, error) {

	fd := dc.Filter(model.Features)
	svm, ok := model.Model.(*svm.SVM)
	if !ok {
		return nil, fmt.Errorf("not an SVM")
	}
	return svm.Predictions(fd)
}

// svmCombinations creates an svm model for each combination of
// the feature vector with respect to the size param.
// It returns an array of svm, one for each combination.
// todo(santiaago): move to ml
//
func svmCombinations(dc data.Container, size int) (models ml.ModelContainers) {

	combs := itertools.Combinations(dc.Features, size)

	for _, c := range combs {
		if *svmK == 1 {
			for k := 1; k <= *svmKRange; k++ {
				fmt.Printf("\r%v/%v", c, len(combs))
				fd := dc.FilterWithPredict(c)
				svm := svm.NewSVM()
				svm.K = k
				svm.Lambda = *svmLambda
				svm.T = *svmT
				svm.InitializeFromData(fd)

				name := fmt.Sprintf("svm 1D %v k %v T %v", c, k, *svmT)

				if err := svm.Learn(); err == nil {
					models = append(models, ml.NewModelContainer(svm, name, c))
				}
			}
		} else {
			fmt.Printf("\r%v/%v", c, len(combs))
			fd := dc.FilterWithPredict(c)
			svm := svm.NewSVM()
			svm.K = *svmK
			svm.Lambda = *svmLambda
			svm.T = *svmT
			svm.InitializeFromData(fd)

			name := fmt.Sprintf("svm 1D %v k %v T %v", c, *svmK, *svmT)

			if err := svm.Learn(); err == nil {
				models = append(models, ml.NewModelContainer(svm, name, c))
			}
		}
	}
	fmt.Println()
	return
}

func specificSvmModels(dc data.Container) (models ml.ModelContainers) {

	cases := []struct {
		features []int
		name     string
	}{
		{
			[]int{passengerIndexSex, passengerIndexAge},
			"svm Sex Age",
		},
		{
			[]int{passengerIndexAge, passengerIndexPclass},
			"svm PClass Age",
		},

		{
			[]int{passengerIndexSex, passengerIndexPclass},
			"svm PClass Sex",
		},
		{
			[]int{passengerIndexSex, passengerIndexAge, passengerIndexPclass},
			"svm Sex Age PClass",
		},
	}

	for _, c := range cases {
		svm := svm.NewSVM()
		fd := dc.FilterWithPredict(c.features)
		svm.InitializeFromData(fd)

		if err := svm.Learn(); err != nil {
			continue
		}
		mc := ml.NewModelContainer(svm, c.name, c.features)
		models = append(models, mc)
	}
	return
}
