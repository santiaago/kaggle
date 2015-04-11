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

// passengerFeatures return an array of indexes of the
// passenger colomns data can be used to learn.
func passengerFeatures() []int {
	return []int{
		passengerIndexPclass,
		passengerIndexSex,
		passengerIndexAge,
		passengerIndexSibSp,
		passengerIndexParch,
		passengerIndexTicket,
		passengerIndexFare,
		passengerIndexCabin,
		passengerIndexEmbarked,
	}
}

func (p passenger) String() (s string) {
	s += fmt.Sprintf("ID %v\n", p.ID)
	s += fmt.Sprintf("Survived %t\n", p.Survived)
	s += fmt.Sprintf("Pclass %v\n", p.Pclass)
	s += fmt.Sprintf("Name %v\n", p.Name)
	s += fmt.Sprintf("Sex %v\n", p.Sex)
	s += fmt.Sprintf("Age %v\n", p.Age)
	s += fmt.Sprintf("SibSp %v\n", p.SibSp)
	s += fmt.Sprintf("Sex %v\n", p.Parch)
	s += fmt.Sprintf("Ticket %v\n", p.Ticket)
	s += fmt.Sprintf("Cabin %v\n", p.Cabin)
	s += fmt.Sprintf("Fare %v\n", p.Fare)
	s += fmt.Sprintf("Cabin %v\n", p.Cabin)
	s += fmt.Sprintf("Embarked %v\n", p.Embarked)
	return
}

func prepareData(passengers []passenger) (data [][]float64) {

	for i := 0; i < len(passengers); i++ {
		p := passengers[i]

		var survived float64 = -1
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

		d := []float64{
			0,
			survived,
			pclass,
			0,
			sex,
			age,
			sibsp,
			parch,
			0,
			fare,
			0,
			embarked,
		}
		data = append(data, d)
	}
	return
}
