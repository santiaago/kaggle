package main

import (
	"fmt"
	"strconv"

	"github.com/santiaago/caltechx.go/linear"
	"github.com/santiaago/caltechx.go/linreg"
)

//
// PassengerId,Survived,
// Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
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

// creates a linear regression model for each combination of
// vector of 3 features and returns an array of linear regressions.
func linregVectorsOf3(passengers []passenger) []*linreg.LinearRegression {
	var data [][]float64
	for i := 0; i < len(passengers); i++ {
		p := passengers[i]

		var survived float64
		if p.Survived {
			survived = float64(1)
		}

		var pclass float64
		if pc, err := strconv.ParseInt(p.Pclass, 10, 32); err != nil {
			pclass = float64(3)
		} else {
			pclass = float64(pc)
		}

		var sex float64
		if p.Sex == "female" {
			sex = float64(1)
		}

		var age = float64(p.Age)

		var sibsp = float64(p.SibSp)
		var parch = float64(p.Parch)
		//var ticket = float64(p.Ticket)
		var fare = float64(p.Fare)
		//var cabin = float64(p.Cabin)

		var embarked float64
		if len(p.Embarked) == 0 {
			embarked = float64(0)
		} else if p.Embarked == "C" {
			embarked = float64(0)
		} else if p.Embarked == "Q" {
			embarked = float64(1)
		} else if p.Embarked == "S" {
			embarked = float64(2)
		}

		d := []float64{sex, pclass, age, sibsp, parch, fare, embarked, survived}
		data = append(data, d)
	}
	var linregs []*linreg.LinearRegression
	combs := combinations([]int{0, 1, 2, 3, 4, 5, 6}, 3)
	for _, comb := range combs {
		filteredData := filter(data, comb)
		linreg := linreg.NewLinearRegression()
		linreg.InitializeFromData(filteredData)
		linreg.Learn()
		fmt.Println("using combination", comb)
		fmt.Printf("EIn = %f\n", linreg.Ein())

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
	var data [][]float64
	for i := 0; i < len(passengers); i++ {
		p := passengers[i]
		var survived float64
		if p.Survived {
			survived = float64(1)
		}

		var sex float64
		if p.Sex == "female" {
			sex = float64(1)
		}

		var pclass float64
		if pc, err := strconv.ParseInt(p.Pclass, 10, 32); err != nil {
			pclass = float64(3)
		} else {
			pclass = float64(pc)
		}

		d := []float64{sex, float64(p.Age), pclass, survived}
		data = append(data, d)
	}
	linreg := linreg.NewLinearRegression()
	linreg.InitializeFromData(data)
	linreg.Learn()
	fmt.Printf("EIn = %f\n", linreg.Ein())
	return linreg
}

func linregSexAge(passengers []passenger) *linreg.LinearRegression {
	var data [][]float64
	for i := 0; i < len(passengers); i++ {
		p := passengers[i]
		var survived float64
		if p.Survived {
			survived = float64(1)
		}

		var sex float64
		if p.Sex == "female" {
			sex = float64(1)
		}
		d := []float64{sex, float64(p.Age), survived}
		data = append(data, d)
	}
	linreg := linreg.NewLinearRegression()
	linreg.InitializeFromData(data)
	linreg.Learn()
	fmt.Printf("EIn = %f\n", linreg.Ein())
	return linreg
}

func linregPClassAge(passengers []passenger) *linreg.LinearRegression {
	var data [][]float64
	for i := 0; i < len(passengers); i++ {
		p := passengers[i]
		var survived float64
		if p.Survived {
			survived = float64(1)
		}
		var pclass float64
		if pc, err := strconv.ParseInt(p.Pclass, 10, 32); err != nil {
			pclass = float64(3)
		} else {
			pclass = float64(pc)
		}

		d := []float64{pclass, float64(p.Age), survived}
		data = append(data, d)
	}
	linreg := linreg.NewLinearRegression()
	linreg.InitializeFromData(data)
	linreg.Learn()
	fmt.Printf("EIn = %f\n", linreg.Ein())
	return linreg
}

func linregPClassSex(passengers []passenger) *linreg.LinearRegression {
	var data [][]float64
	for i := 0; i < len(passengers); i++ {
		p := passengers[i]

		var survived float64
		if p.Survived {
			survived = float64(1)
		}

		var pclass float64
		if pc, err := strconv.ParseInt(p.Pclass, 10, 32); err != nil {
			pclass = float64(3)
		} else {
			pclass = float64(pc)
		}

		var sex float64
		if p.Sex == "female" {
			sex = float64(1)
		}
		d := []float64{sex, pclass, survived}
		data = append(data, d)
	}
	linreg := linreg.NewLinearRegression()
	linreg.InitializeFromData(data)
	linreg.Learn()
	fmt.Printf("number of passengers: %d\n", len(passengers))
	fmt.Printf("EIn = %f\n", linreg.Ein())
	return linreg
}
