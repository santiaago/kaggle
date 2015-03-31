package main

import (
	"fmt"
	"strconv"
)

const (
	passengerIndexID int = iota
	passengerIndexSurvived
	passengerIndexPclass
	passengerIndexName
	passengerIndexSex
	passengerIndexAge
	passengerIndexSibSp
	passengerIndexParch
	passengerIndexTicket
	passengerIndexFare
	passengerIndexCabin
	passengerIndexEmbarked
)

type passenger struct {
	ID       string
	Survived bool
	Pclass   string
	Name     string
	Sex      string
	Age      int
	SibSp    int
	Parch    int
	Ticket   string
	Fare     int
	Cabin    string
	Embarked string
}

func (p passenger) Print() {
	fmt.Println("ID", p.ID)
	fmt.Println("Survived", p.Survived)
	fmt.Println("Pclass", p.Pclass)
	fmt.Println("Name", p.Name)
	fmt.Println("Sex", p.Sex)
	fmt.Println("Age", p.Age)
	fmt.Println("SibSp", p.SibSp)
	fmt.Println("Sex", p.Parch)
	fmt.Println("Ticket", p.Ticket)
	fmt.Println("Cabin", p.Cabin)
	fmt.Println("Fare", p.Fare)
	fmt.Println("Cabin", p.Cabin)
	fmt.Println("Embarked", p.Embarked)
}

/*func prepareTestData(passengers *[]passenger) (data [][]float64) {
	var data [][]float64
	for i := 0; i < len(*passengers); i++ {
		p := (*passengers)[i]

		var sex float64
		if p.Sex == "female" {
			sex = float64(1)
		}
		survived := float64(0)
		d := []float64{sex, float64(p.Age)}
		data = append(data, d)
	}

}*/

func prepareData(passengers []passenger) (data [][]float64) {

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
	return
}
