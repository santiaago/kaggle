package main

import (
	"fmt"

	"github.com/santiaago/caltechx.go/linear"
	"github.com/santiaago/caltechx.go/linreg"
)

//
// PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
//

// linregTest sets the Survived field in each passenger of the passenger array
// with respect to the linear regression 'linreg' passed as argument.
func linregTest(linreg *linreg.LinearRegression, passengers *[]passenger, keep []int) {

	data := prepareData(*passengers)
	filteredData := filter(data, keep)

	for i := 0; i < len(filteredData); i++ {
		oX := filteredData[i]
		gi := float64(0)
		for j := 0; j < len(oX); j++ {
			gi += oX[j] * linreg.Wn[j]
		}

		if linear.Sign(gi) > 0 {
			data[i][passengerIndexSurvived] = 1 // do we need this?
			(*passengers)[i].Survived = true
		}
	}
}

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

// creates a linear regression model for each combination of
// vector of 3 features and returns an array of linear regressions.
func linregVectors(passengers []passenger, size int) (linregs []*linreg.LinearRegression, usedFeatures [][]int) {

	data := prepareData(passengers)

	combs := combinations([]int{0, 1, 2, 3, 4, 5, 6}, size)
	for _, comb := range combs {
		filteredData := filter(data, comb)
		linreg := linreg.NewLinearRegression()
		linreg.Name = fmt.Sprintf("LinregModel-V-%d-%v", size, comb)
		linreg.InitializeFromData(filteredData)
		linreg.Learn()
		fmt.Printf("EIn = %f \t using combination %v\n", linreg.Ein(), comb)

		usedFeatures = append(usedFeatures, comb)
		linregs = append(linregs, linreg)
	}
	return
}

func filter(data [][]float64, keep []int) [][]float64 {
	var filtered [][]float64
	for i := 0; i < len(data); i++ {
		var df []float64
		for j := 0; j < len(keep); j++ {
			df = append(df, data[i][keep[j]])
		}
		df = append(df, data[i][len(data[i])-1])
		filtered = append(filtered, df)
	}
	return filtered
}

func linregSexAgePClass(passengers []passenger) (lr *linreg.LinearRegression, usedFeatures []int) {

	data := prepareData(passengers)
	usedFeatures = []int{passengerIndexSex, passengerIndexAge, passengerIndexPclass}
	filteredData := filter(data, usedFeatures)
	lr = linreg.NewLinearRegression()
	lr.Name = "Sex Age PClass"
	lr.InitializeFromData(filteredData)
	lr.Learn()
	fmt.Printf("EIn = %f \t%s\tfeatures used %v\n", lr.Ein(), lr.Name, usedFeatures)
	return lr, usedFeatures
}

func linregSexAge(passengers []passenger) (lr *linreg.LinearRegression, usedFeatures []int) {
	data := prepareData(passengers)
	usedFeatures = []int{passengerIndexSex, passengerIndexAge}
	filteredData := filter(data, usedFeatures)
	lr = linreg.NewLinearRegression()
	lr.Name = "Sex Age"
	lr.InitializeFromData(filteredData)
	lr.Learn()
	fmt.Printf("EIn = %f \t%s\tfeatures used %v\n", lr.Ein(), lr.Name, usedFeatures)
	return lr, usedFeatures
}

func linregPClassAge(passengers []passenger) (lr *linreg.LinearRegression, usedFeatures []int) {
	data := prepareData(passengers)
	usedFeatures = []int{passengerIndexAge, passengerIndexPclass}
	filteredData := filter(data, usedFeatures)
	lr = linreg.NewLinearRegression()
	lr.Name = "PClass Age"
	lr.InitializeFromData(filteredData)
	lr.Learn()
	fmt.Printf("EIn = %f \t%s\tfeatures used %v\n", lr.Ein(), lr.Name, usedFeatures)
	return lr, usedFeatures
}

func linregPClassSex(passengers []passenger) (lr *linreg.LinearRegression, usedFeatures []int) {
	data := prepareData(passengers)
	usedFeatures = []int{passengerIndexSex, passengerIndexPclass}
	filteredData := filter(data, usedFeatures)

	lr = linreg.NewLinearRegression()
	lr.Name = "PClass Sex"
	lr.InitializeFromData(filteredData)
	lr.Learn()
	fmt.Printf("EIn = %f \t%s\tfeatures used %v\n", lr.Ein(), lr.Name, usedFeatures)
	return lr, usedFeatures
}
