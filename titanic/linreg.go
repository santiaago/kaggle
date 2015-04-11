package main

import (
	"fmt"

	"github.com/santiaago/caltechx.go/linear"
	"github.com/santiaago/caltechx.go/linreg"
	"github.com/santiaago/kaggle/data"
	"github.com/santiaago/kaggle/itertools"
)

// linregTest sets the Survived field of each passenger in the passenger array
// with respect to the prediction set by the linear regression 'linreg' passed as argument.
func linregTest(lr *linreg.LinearRegression, dc data.Container, keep []int) (predictions []int) {
	fd := filter(dc.Data, keep)
	for i := 0; i < len(fd); i++ {

		x := []float64{1}
		x = append(x, fd[i][:len(fd[i])-1]...)

		if lr.UsesTranformFunction {
			x = lr.TransformFunction(x)
		}

		gi := prediction(lr, x)

		if linear.Sign(gi) == 1 {
			predictions = append(predictions, 1)
		} else {
			predictions = append(predictions, 0)
		}
	}
	return
}

// prediction returns the result of the dot product between the x vector passed as param
// and the linear regression vector of weights.
// todo(santiaago): move this to caltechx.go
func prediction(lr *linreg.LinearRegression, x []float64) (p float64) {
	for j := 0; j < len(x); j++ {
		p += x[j] * lr.Wn[j]
	}
	return
}

// linregVectorsOfInterval returns an array functions.
// These functions return an array of linear regression and the corresponding features used.
func linregAllCombinations() (funcs []func(data.Container) ([]*linreg.LinearRegression, [][]int)) {
	funcs = []func(dc data.Container) ([]*linreg.LinearRegression, [][]int){
		func(dc data.Container) ([]*linreg.LinearRegression, [][]int) {
			return linregCombinations(dc, 2)
		},
		func(dc data.Container) ([]*linreg.LinearRegression, [][]int) {
			return linregCombinations(dc, 3)
		},
		func(dc data.Container) ([]*linreg.LinearRegression, [][]int) {
			return linregCombinations(dc, 4)
		},
		func(dc data.Container) ([]*linreg.LinearRegression, [][]int) {
			return linregCombinations(dc, 5)
		},
		func(dc data.Container) ([]*linreg.LinearRegression, [][]int) {
			return linregCombinations(dc, 6)
		},
		func(dc data.Container) ([]*linreg.LinearRegression, [][]int) {
			return linregCombinations(dc, 7)
		},
	}
	return
}

// linregCombinations creates a linear regression model for each combination of
// the feature vector with respect to the size param.
// It returns an array of linear regressions, one for each combination.
func linregCombinations(dc data.Container, size int) (lrs []*linreg.LinearRegression, features [][]int) {

	combs := itertools.Combinations(dc.Features, size)

	for _, c := range combs {
		fd := filter(dc.Data, c)
		lr := linreg.NewLinearRegression()
		lr.InitializeFromData(fd)

		lr.Name = fmt.Sprintf("LinregModel-V-%d-%v", size, c)

		if err := lr.Learn(); err == nil {
			fmt.Printf("EIn = %f \t using combination %v\n", lr.Ein(), c)

			features = append(features, c)
			lrs = append(lrs, lr)
		}
	}
	return
}

// filter returns the same data passed as param filtered with respect to the keep array.
// the keep array in an array of the indexes to keep in the data.
func filter(data [][]float64, keep []int) (filtered [][]float64) {
	for i := 0; i < len(data); i++ {
		var row []float64
		for j := 0; j < len(keep); j++ {
			row = append(row, data[i][keep[j]])
		}
		// todo(santiaago): handle passengerIndexSurvived differently
		row = append(row, data[i][passengerIndexSurvived])
		filtered = append(filtered, row)
	}
	return
}

func specificLinregFuncs() []func(dc data.Container) (*linreg.LinearRegression, []int) {
	return []func(dc data.Container) (*linreg.LinearRegression, []int){
		linregSexAge,
		linregPClassAge,
		linregPClassSex,
		linregSexAgePClass,
	}
}

func linregSexAgePClass(dc data.Container) (lr *linreg.LinearRegression, features []int) {
	features = []int{
		passengerIndexSex,
		passengerIndexAge,
		passengerIndexPclass,
	}

	fd := filter(dc.Data, features)

	lr = linreg.NewLinearRegression()
	lr.Name = "Sex Age PClass"
	lr.InitializeFromData(fd)
	if err := lr.Learn(); err == nil {
		fmt.Printf("EIn = %f \t%s\tfeatures used %v\n", lr.Ein(), lr.Name, features)
		return
	}
	return nil, nil
}

func linregSexAge(dc data.Container) (lr *linreg.LinearRegression, features []int) {

	features = []int{
		passengerIndexSex,
		passengerIndexAge,
	}

	fd := filter(dc.Data, features)

	lr = linreg.NewLinearRegression()
	lr.InitializeFromData(fd)

	lr.Name = "Sex Age"
	if err := lr.Learn(); err == nil {
		fmt.Printf("EIn = %f \t%s\tfeatures used %v\n", lr.Ein(), lr.Name, features)
		return
	}
	return nil, nil
}

func linregPClassAge(dc data.Container) (lr *linreg.LinearRegression, features []int) {

	features = []int{
		passengerIndexAge,
		passengerIndexPclass,
	}

	fd := filter(dc.Data, features)

	lr = linreg.NewLinearRegression()
	lr.InitializeFromData(fd)

	lr.Name = "PClass Age"
	if err := lr.Learn(); err == nil {
		fmt.Printf("EIn = %f \t%s\tfeatures used %v\n", lr.Ein(), lr.Name, features)
		return
	}
	return nil, nil
}

func linregPClassSex(dc data.Container) (lr *linreg.LinearRegression, features []int) {

	features = []int{
		passengerIndexSex,
		passengerIndexPclass,
	}

	fd := filter(dc.Data, features)

	lr = linreg.NewLinearRegression()
	lr.InitializeFromData(fd)

	lr.Name = "PClass Sex"
	if err := lr.Learn(); err == nil {
		fmt.Printf("EIn = %f \t%s\tfeatures used %v\n", lr.Ein(), lr.Name, features)
		return
	}
	return nil, nil
}
