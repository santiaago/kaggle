package main

import (
	"fmt"
	"strconv"

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
	fmt.Printf("number of passengers: %d\n", len(passengers))
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
	fmt.Printf("number of passengers: %d\n", len(passengers))
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
