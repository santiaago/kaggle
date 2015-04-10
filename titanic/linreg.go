package main

import (
	"fmt"

	"github.com/santiaago/caltechx.go/linear"
	"github.com/santiaago/caltechx.go/linreg"
)

// linregTest sets the Survived field of each passenger in the passenger array
// with respect to the prediction set by the linear regression 'linreg' passed as argument.
func linregTest(linreg *linreg.LinearRegression, passengers *[]passenger, keep []int) {

	data := prepareData(*passengers)
	filteredData := filter(data, keep)
	for i := 0; i < len(filteredData); i++ {
		oX := []float64{1}
		oX = append(oX, filteredData[i][:len(filteredData[i])-1]...)
		if linreg.UsesTranformFunction {
			oX = linreg.TransformFunction(oX)
		}

		gi := prediction(linreg, oX)

		if linear.Sign(gi) == 1 {
			(*passengers)[i].Survived = true
		}
	}
}

// prediction returns the result of the dot product between the x vector passed as param
// and the linear regression vector of weights.
// todo(santiaago): move this to caltechx.go
func prediction(linreg *linreg.LinearRegression, x []float64) (p float64) {
	for j := 0; j < len(x); j++ {
		p += x[j] * linreg.Wn[j]
	}
	return
}

// linregVectorsOfInterval returns an array functions.
// These functions return an array of linear regression and the corresponding features used.
// todo(santiaago): rename function.
func linregVectorsOfInterval() (funcs []func([]passenger) ([]*linreg.LinearRegression, [][]int)) {
	funcs = []func(ps []passenger) ([]*linreg.LinearRegression, [][]int){
		func(ps []passenger) ([]*linreg.LinearRegression, [][]int) {
			return linregVectors(ps, 2)
		},
		func(ps []passenger) ([]*linreg.LinearRegression, [][]int) {
			return linregVectors(ps, 3)
		},
		func(ps []passenger) ([]*linreg.LinearRegression, [][]int) {
			return linregVectors(ps, 4)
		},
		func(ps []passenger) ([]*linreg.LinearRegression, [][]int) {
			return linregVectors(ps, 5)
		},
		func(ps []passenger) ([]*linreg.LinearRegression, [][]int) {
			return linregVectors(ps, 6)
		},
		func(ps []passenger) ([]*linreg.LinearRegression, [][]int) {
			return linregVectors(ps, 7)
		},
	}
	return
}

// linregVectors creates a linear regression model for each combination of
// vector of x features and returns an array of linear regressions for each combination.
func linregVectors(passengers []passenger, size int) (linregs []*linreg.LinearRegression, usedFeatures [][]int) {

	data := prepareData(passengers)

	features := []int{
		passengerIndexPclass,
		passengerIndexName,
		passengerIndexSex,
		passengerIndexAge,
		passengerIndexSibSp,
		passengerIndexParch,
		passengerIndexTicket,
		passengerIndexFare,
		passengerIndexCabin,
		passengerIndexEmbarked,
	}

	combs := combinations(features, size)
	for _, comb := range combs {
		filteredData := filter(data, comb)
		linreg := linreg.NewLinearRegression()
		linreg.Name = fmt.Sprintf("LinregModel-V-%d-%v", size, comb)
		linreg.InitializeFromData(filteredData)
		if err := linreg.Learn(); err == nil {
			fmt.Printf("EIn = %f \t using combination %v\n", linreg.Ein(), comb)

			usedFeatures = append(usedFeatures, comb)
			linregs = append(linregs, linreg)
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
		row = append(row, data[i][passengerIndexSurvived])
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

func linregSexAgePClass(passengers []passenger) (lr *linreg.LinearRegression, usedFeatures []int) {
	data := prepareData(passengers)

	usedFeaturesInternal := []int{
		passengerIndexSex,
		passengerIndexAge,
		passengerIndexPclass,
	}
	usedFeatures = usedFeaturesInternal[:3]
	filteredData := filter(data, usedFeaturesInternal)

	lr = linreg.NewLinearRegression()
	lr.Name = "Sex Age PClass"
	lr.InitializeFromData(filteredData)
	if err := lr.Learn(); err == nil {
		fmt.Printf("EIn = %f \t%s\tfeatures used %v\n", lr.Ein(), lr.Name, usedFeatures)
		return
	}
	return nil, nil
}

func linregSexAge(passengers []passenger) (lr *linreg.LinearRegression, usedFeatures []int) {
	data := prepareData(passengers)
	usedFeaturesInternal := []int{
		passengerIndexSex,
		passengerIndexAge,
	}
	usedFeatures = usedFeaturesInternal[:2]

	filteredData := filter(data, usedFeaturesInternal)
	lr = linreg.NewLinearRegression()
	lr.Name = "Sex Age"
	lr.InitializeFromData(filteredData)
	if err := lr.Learn(); err == nil {
		fmt.Printf("EIn = %f \t%s\tfeatures used %v\n", lr.Ein(), lr.Name, usedFeatures)
		return
	}
	return nil, nil
}

func linregPClassAge(passengers []passenger) (lr *linreg.LinearRegression, usedFeatures []int) {
	data := prepareData(passengers)

	usedFeaturesInternal := []int{
		passengerIndexAge,
		passengerIndexPclass,
	}
	usedFeatures = usedFeaturesInternal[:2]

	filteredData := filter(data, usedFeaturesInternal)
	lr = linreg.NewLinearRegression()
	lr.Name = "PClass Age"
	lr.InitializeFromData(filteredData)
	if err := lr.Learn(); err == nil {
		fmt.Printf("EIn = %f \t%s\tfeatures used %v\n", lr.Ein(), lr.Name, usedFeatures)
		return
	}
	return nil, nil
}

func linregPClassSex(passengers []passenger) (lr *linreg.LinearRegression, usedFeatures []int) {
	data := prepareData(passengers)

	usedFeaturesInternal := []int{
		passengerIndexSex,
		passengerIndexPclass,
	}
	usedFeatures = usedFeaturesInternal[:2]

	filteredData := filter(data, usedFeaturesInternal)
	lr = linreg.NewLinearRegression()
	lr.Name = "PClass Sex"
	lr.InitializeFromData(filteredData)
	if err := lr.Learn(); err == nil {
		fmt.Printf("EIn = %f \t%s\tfeatures used %v\n", lr.Ein(), lr.Name, usedFeatures)
		return
	}
	return nil, nil
}
