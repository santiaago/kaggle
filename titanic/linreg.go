package main

import (
	"fmt"

	"github.com/santiaago/caltechx.go/linear"
	"github.com/santiaago/caltechx.go/linreg"
)

// linregTest sets the Survived field in each passenger of the passenger array
// with respect to the linear regression 'linreg' passed as argument.
func linregTest(linreg *linreg.LinearRegression, passengers *[]passenger, keep []int) {

	data := prepareData(*passengers)
	filteredData := filter(data, keep)
	for i := 0; i < len(filteredData); i++ {
		oX := []float64{1}
		oX = append(oX, filteredData[i][:len(filteredData[i])-1]...)
		if linreg.UsesTranformFunction {
			oX = linreg.TransformFunction(oX)
		}
		gi := float64(0)
		for j := 0; j < len(oX); j++ {
			gi += oX[j] * linreg.Wn[j]
		}

		if linear.Sign(gi) > 0 {
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

	toTry := []int{
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

	combs := combinations(toTry, size)
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

func filter(data [][]float64, keep []int) [][]float64 {
	var filtered [][]float64
	for i := 0; i < len(data); i++ {
		var df []float64
		for j := 0; j < len(keep); j++ {
			df = append(df, data[i][keep[j]])
		}
		df = append(df, data[i][passengerIndexSurvived])
		filtered = append(filtered, df)
	}
	return filtered
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
