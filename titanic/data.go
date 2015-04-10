package main

import (
	"encoding/csv"
	"fmt"
	"strconv"
)

type Reader struct {
	ex Extractor
	cl Cleaner
}

func (r Reader) Read() ([][]float64, error) {
	d, err := r.ex.Extract()
	if err != nil {
		return nil, fmt.Errorf("error when extracting data: %v", err)
	}
	return r.cl.Clean(d)
}

type Extractor interface {
	Extract() (interface{}, error)
}

// A PassengerExtractor extract passenger data from a CVS-encoded file.
type PassengerExtractor struct {
	r *csv.Reader
}

// NewPassengerExtractor returns a nez PassengerExtractor that extracts from r.
func NewPassengerExtractor(r *csv.Reader) PassengerExtractor {
	return PassengerExtractor{r}
}

// Extract extracts the  passengers from r.
func (pex PassengerExtractor) Extract() (interface{}, error) {
	return passengersFromTrainingSet(pex.r), nil
}

type Cleaner interface {
	Clean(interface{}) ([][]float64, error)
}

type PassengerCleaner struct {
}

func NewPassengerCleaner() PassengerCleaner {
	return PassengerCleaner{}
}

func (pc PassengerCleaner) Clean(passengers interface{}) ([][]float64, error) {

	if ps, ok := passengers.([]passenger); ok {
		return prepareData(ps), nil
	}
	return nil, fmt.Errorf("unable to clean unknown type.")
}

// passengerFromTrainingRow creates a passenger object
// from a train data row.
// A training row looks like this:
//     PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
func passengerFromTrainingRow(line []string) passenger {

	survived, err := strconv.ParseBool(line[1])
	if err != nil {
		survived = false
	}

	age, err := strconv.ParseInt(line[5], 10, 32)
	if err != nil {
		age = 25
	}

	sibsp, err := strconv.ParseInt(line[6], 10, 32)
	if err != nil {
		sibsp = 0
	}

	parch, err := strconv.ParseInt(line[7], 10, 32)
	if err != nil {
		parch = 0
	}

	fare, err := strconv.ParseInt(line[9], 10, 32)
	if err != nil {
		fare = 0
	}

	var embarked string
	if len(line) > 11 {
		embarked = line[11]
	}

	p := passenger{
		line[0],
		survived,
		line[2],
		line[3],
		line[4],
		int(age),
		int(sibsp),
		int(parch),
		line[8],
		int(fare),
		line[10],
		embarked,
	}
	return p
}

// passengerFromTestingRow creates a passenger object
// from a test data row.
func passengerFromTestingRow(line []string) passenger {
	id := line[0]
	pclass := line[1]
	name := line[2]
	sex := line[3]
	age, err := strconv.ParseInt(line[4], 10, 32)
	if err != nil {
		age = 33
	}
	sibsp, err := strconv.ParseInt(line[5], 10, 32)
	if err != nil {
		sibsp = 0
	}
	parch, err := strconv.ParseInt(line[6], 10, 32)
	if err != nil {
		parch = 0
	}
	ticket := line[7]
	fare, err := strconv.ParseInt(line[8], 10, 32)
	if err != nil {
		fare = 0
	}

	cabin := line[9]
	var embarked string
	if len(line) > 10 {
		embarked = line[10]
	}

	p := passenger{
		id,
		false,
		pclass,
		name,
		sex,
		int(age),
		int(sibsp),
		int(parch),
		ticket,
		int(fare),
		cabin,
		embarked,
	}
	return p
}
