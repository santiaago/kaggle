package main

import (
	"fmt"

	"github.com/santiaago/caltechx.go/linear"
	"github.com/santiaago/caltechx.go/linreg"
)

// linregTest sets the Survived field of each passenger in the passenger array
// with respect to the prediction set by the linear regression 'linreg' passed as argument.
func linregTest(lr *linreg.LinearRegression, passengers *[]passenger, keep []int) {
	d := prepareData(*passengers)
	fd := filter(d, keep)
	for i := 0; i < len(fd); i++ {

		x := []float64{1}
		x = append(x, fd[i][:len(fd[i])-1]...)

		if lr.UsesTranformFunction {
			x = lr.TransformFunction(x)
		}

		gi := prediction(lr, x)

		if linear.Sign(gi) == 1 {
			(*passengers)[i].Survived = true
		}
	}
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
func linregAllCombinations() (funcs []func([]passenger) ([]*linreg.LinearRegression, [][]int)) {
	funcs = []func(ps []passenger) ([]*linreg.LinearRegression, [][]int){
		func(ps []passenger) ([]*linreg.LinearRegression, [][]int) {
			return linregCombinations(ps, 2)
		},
		func(ps []passenger) ([]*linreg.LinearRegression, [][]int) {
			return linregCombinations(ps, 3)
		},
		func(ps []passenger) ([]*linreg.LinearRegression, [][]int) {
			return linregCombinations(ps, 4)
		},
		func(ps []passenger) ([]*linreg.LinearRegression, [][]int) {
			return linregCombinations(ps, 5)
		},
		func(ps []passenger) ([]*linreg.LinearRegression, [][]int) {
			return linregCombinations(ps, 6)
		},
		func(ps []passenger) ([]*linreg.LinearRegression, [][]int) {
			return linregCombinations(ps, 7)
		},
	}
	return
}

// linregCombinations creates a linear regression model for each combination of
// the feature vector with respect to the size param.
// It returns an array of linear regressions, one for each combination.
func linregCombinations(passengers []passenger, size int) (lrs []*linreg.LinearRegression, features [][]int) {

	d := prepareData(passengers)
	f := passengerFeatures()
	combs := combinations(f, size)

	for _, c := range combs {
		fd := filter(d, c)
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
func filter(d [][]float64, keep []int) (filtered [][]float64) {
	for i := 0; i < len(d); i++ {
		var row []float64
		for j := 0; j < len(keep); j++ {
			row = append(row, d[i][keep[j]])
		}
		row = append(row, d[i][passengerIndexSurvived])
		filtered = append(filtered, row)
	}
	return
}

func specificLinregFuncs() []func(passengers []passenger) (*linreg.LinearRegression, []int) {
	return []func(passengers []passenger) (*linreg.LinearRegression, []int){
		linregSexAge,
		linregPClassAge,
		linregPClassSex,
		linregSexAgePClass,
	}
}

func linregSexAgePClass(passengers []passenger) (lr *linreg.LinearRegression, features []int) {

	featuresInternal := []int{
		passengerIndexSex,
		passengerIndexAge,
		passengerIndexPclass,
	}
	features = featuresInternal[:3]

	d := prepareData(passengers)
	fd := filter(d, featuresInternal)

	lr = linreg.NewLinearRegression()
	lr.Name = "Sex Age PClass"
	lr.InitializeFromData(fd)
	if err := lr.Learn(); err == nil {
		fmt.Printf("EIn = %f \t%s\tfeatures used %v\n", lr.Ein(), lr.Name, features)
		return
	}
	return nil, nil
}

func linregSexAge(passengers []passenger) (lr *linreg.LinearRegression, features []int) {

	featuresInternal := []int{
		passengerIndexSex,
		passengerIndexAge,
	}
	features = featuresInternal[:2]

	d := prepareData(passengers)
	fd := filter(d, featuresInternal)

	lr = linreg.NewLinearRegression()
	lr.InitializeFromData(fd)

	lr.Name = "Sex Age"
	if err := lr.Learn(); err == nil {
		fmt.Printf("EIn = %f \t%s\tfeatures used %v\n", lr.Ein(), lr.Name, features)
		return
	}
	return nil, nil
}

func linregPClassAge(passengers []passenger) (lr *linreg.LinearRegression, features []int) {

	featuresInternal := []int{
		passengerIndexAge,
		passengerIndexPclass,
	}
	features = featuresInternal[:2]

	d := prepareData(passengers)
	fd := filter(d, featuresInternal)

	lr = linreg.NewLinearRegression()
	lr.InitializeFromData(fd)

	lr.Name = "PClass Age"
	if err := lr.Learn(); err == nil {
		fmt.Printf("EIn = %f \t%s\tfeatures used %v\n", lr.Ein(), lr.Name, features)
		return
	}
	return nil, nil
}

func linregPClassSex(passengers []passenger) (lr *linreg.LinearRegression, features []int) {

	featuresInternal := []int{
		passengerIndexSex,
		passengerIndexPclass,
	}
	features = featuresInternal[:2]

	d := prepareData(passengers)
	fd := filter(d, featuresInternal)

	lr = linreg.NewLinearRegression()
	lr.InitializeFromData(fd)

	lr.Name = "PClass Sex"
	if err := lr.Learn(); err == nil {
		fmt.Printf("EIn = %f \t%s\tfeatures used %v\n", lr.Ein(), lr.Name, features)
		return
	}
	return nil, nil
}
