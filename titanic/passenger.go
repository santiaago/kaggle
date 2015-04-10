package main

import (
	"encoding/csv"
	"fmt"
	"log"
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

func passengersFromTrainingSet(r *csv.Reader) (passengers []passenger) {
	var rawData [][]string
	var err error
	if rawData, err = r.ReadAll(); err != nil {
		log.Fatalln(err)
		return nil
	}

	for i := 1; i < len(rawData); i++ {
		p := passengerFromTrainingRow(rawData[i])
		passengers = append(passengers, p)
	}
	return
}

func prepareData(passengers []passenger) (data [][]float64) {

	for i := 0; i < len(passengers); i++ {
		p := passengers[i]

		var survived float64
		if p.Survived {
			survived = float64(1)
		} else {
			survived = float64(-1)
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
