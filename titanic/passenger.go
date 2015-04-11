package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/santiaago/kaggle/data"
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

// A PassengerReader extract passenger data from a CVS-encoded file.
// It implements the data.Reader interface.
type PassengerReader struct {
	r  *csv.Reader // a CSV encoded reader.
	ex data.Extractor
}

// NewPassengerReader returns a new data.Reader that can read from a given file.
func NewPassengerReader(file string, ex data.Extractor) PassengerReader {
	var r *csv.Reader
	if csvfile, err := os.Open(file); err != nil {
		log.Fatalln(err)
	} else {
		r = csv.NewReader(csvfile)
	}
	return PassengerReader{r, ex}
}

// Clean cleans an array of passengers returning the
// data in the form of 2 dimentional array of float64
func (pr PassengerReader) Clean(passengers interface{}) ([][]float64, error) {

	if ps, ok := passengers.([]passenger); ok {
		return prepareData(ps), nil
	}
	return nil, fmt.Errorf("unable to clean unknown type.")
}

// Read reads the data holded in the csv.Reader r
// by extracting it using the Extract function,
// then Cleans the data and returns it.
func (pr PassengerReader) Read() (data.Container, error) {
	d, err := pr.ex.Extract(pr.r)
	if err != nil {
		return data.Container{}, fmt.Errorf("error when extracting data: %v", err)
	}

	c, err := pr.Clean(d)
	return data.Container{c, passengerFeatures()}, err
}
