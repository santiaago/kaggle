package main

import (
	"fmt"

	"github.com/santiaago/caltechx.go/linear"
	"github.com/santiaago/caltechx.go/linreg"
)

//
// PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
//
func linregTest(linreg *linreg.LinearRegression, passengers *[]passenger) {
	var data [][]float64
	for i := 0; i < len(*passengers); i++ {
		p := (*passengers)[i]

		var sex float64
		if p.Sex == "female" {
			sex = float64(1)
		}
		d := []float64{sex, float64(p.Age), float64(0)}
		data = append(data, d)
	}

	for i := 0; i < len(data); i++ {
		oX := data[i][:1]
		gi := float64(0)
		for j := 0; j < len(oX); j++ {
			gi += oX[j] * linreg.Wn[j]
		}

		if linear.Sign(gi) > 0 {
			data[i][2] = 1
			(*passengers)[i].Survived = true
		}
	}
}

func linregVectorsOfInterval() (funcs []func([]passenger) []*linreg.LinearRegression) {
	funcs = []func(ps []passenger) []*linreg.LinearRegression{
		func(ps []passenger) []*linreg.LinearRegression {
			return linregVectors(ps, 2)
		},
		func(ps []passenger) []*linreg.LinearRegression {
			return linregVectors(ps, 3)
		},
		func(ps []passenger) []*linreg.LinearRegression {
			return linregVectors(ps, 4)
		},
		func(ps []passenger) []*linreg.LinearRegression {
			return linregVectors(ps, 5)
		},
		func(ps []passenger) []*linreg.LinearRegression {
			return linregVectors(ps, 6)
		},
		func(ps []passenger) []*linreg.LinearRegression {
			return linregVectors(ps, 7)
		},
	}
	return
}

// creates a linear regression model for each combination of
// vector of 3 features and returns an array of linear regressions.
func linregVectors(passengers []passenger, size int) []*linreg.LinearRegression {

	data := prepareData(passengers)
	var linregs []*linreg.LinearRegression
	combs := combinations([]int{0, 1, 2, 3, 4, 5, 6}, size)
	for _, comb := range combs {
		filteredData := filter(data, comb)
		linreg := linreg.NewLinearRegression()
		linreg.Name = fmt.Sprintf("LinregModel-V-%d-%v", size, comb)
		linreg.InitializeFromData(filteredData)
		linreg.Learn()
		fmt.Printf("EIn = %f \t using combination %v\n", linreg.Ein(), comb)

		linregs = append(linregs, linreg)
	}
	return linregs
}

func filter(data [][]float64, combination []int) [][]float64 {
	var filtered [][]float64
	for i := 0; i < len(data); i++ {
		var df []float64
		for j := 0; j < len(combination); j++ {
			df = append(df, data[i][combination[j]])
		}
		df = append(df, data[i][len(data[i])-1])
		filtered = append(filtered, df)
	}
	return filtered
}

func linregSexAgePClass(passengers []passenger) *linreg.LinearRegression {

	data := prepareData(passengers)
	filteredData := filter(data, []int{passengerIndexSex, passengerIndexAge, passengerIndexPclass})
	linreg := linreg.NewLinearRegression()
	linreg.Name = "Sex Age PClass"
	linreg.InitializeFromData(filteredData)
	linreg.Learn()
	fmt.Printf("EIn = %f \t%s\n", linreg.Ein(), linreg.Name)
	return linreg
}

func linregSexAge(passengers []passenger) *linreg.LinearRegression {
	data := prepareData(passengers)
	filteredData := filter(data, []int{passengerIndexSex, passengerIndexAge})
	linreg := linreg.NewLinearRegression()
	linreg.Name = "Sex Age"
	linreg.InitializeFromData(filteredData)
	linreg.Learn()
	fmt.Printf("EIn = %f \t%s\n", linreg.Ein(), linreg.Name)
	return linreg
}

func linregPClassAge(passengers []passenger) *linreg.LinearRegression {
	data := prepareData(passengers)
	filteredData := filter(data, []int{passengerIndexAge, passengerIndexPclass})
	linreg := linreg.NewLinearRegression()
	linreg.Name = "PClass Age"
	linreg.InitializeFromData(filteredData)
	linreg.Learn()
	fmt.Printf("EIn = %f \t%s\n", linreg.Ein(), linreg.Name)
	return linreg
}

func linregPClassSex(passengers []passenger) *linreg.LinearRegression {
	data := prepareData(passengers)
	filteredData := filter(data, []int{passengerIndexSex, passengerIndexPclass})

	linreg := linreg.NewLinearRegression()
	linreg.Name = "PClass Sex"
	linreg.InitializeFromData(filteredData)
	linreg.Learn()
	fmt.Printf("EIn = %f \t%s\n", linreg.Ein(), linreg.Name)
	return linreg
}
