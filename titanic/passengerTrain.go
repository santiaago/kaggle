package main

import (
	"encoding/csv"
	"log"
	"strconv"
)

type PassengerTrainExtractor struct{}

func NewPassengerTrainExtractor() PassengerTrainExtractor {
	return PassengerTrainExtractor{}
}

func (pex PassengerTrainExtractor) Extract(r *csv.Reader) (interface{}, error) {
	var rawData [][]string
	var err error
	if rawData, err = r.ReadAll(); err != nil {
		log.Fatalln(err)
		return nil, nil
	}
	passengers := []passenger{}
	for i := 1; i < len(rawData); i++ {
		p := passengerFromTrainingRow(rawData[i])
		passengers = append(passengers, p)
	}
	return passengers, nil
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
