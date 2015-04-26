package main

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"
)

// PassengerTestExtractor type defines how to extract the test data
// and create a passenger type from it.
type PassengerTestExtractor struct{}

// NewPassengerTestExtractor creates a passenger extractor type.
func NewPassengerTestExtractor() PassengerTestExtractor {
	return PassengerTestExtractor{}
}

// Extract returns an array of passengers.
// It extracts them by reading the reader 'r' passed in.
func (pex PassengerTestExtractor) Extract(r *csv.Reader) (interface{}, error) {
	var rawData [][]string
	var err error
	if rawData, err = r.ReadAll(); err != nil {
		log.Fatalln(err)
	}
	passengers := []passenger{}
	for i := 1; i < len(rawData); i++ {
		p := passengerFromTestingRow(rawData[i])
		passengers = append(passengers, p)
	}
	return passengers, nil
}

// PassengerTextWriter type defines how to write the passenger
// data into a file.
type PassengerTestWriter struct {
	passengers []passenger
	ex         PassengerTestExtractor
}

// NewPassengerTestWriter returns a PassengerTestWriter after
// it extracts the file passed in.
func NewPassengerTestWriter(file string) PassengerTestWriter {
	ptw := PassengerTestWriter{}
	reader := NewPassengerReader(file, NewPassengerTestExtractor())

	if data, err := reader.ex.Extract(reader.r); err == nil {
		if ps, ok := data.([]passenger); ok {
			ptw.passengers = ps
		}
	} else {
		log.Println(err)
	}
	return ptw
}

// Write will write to a file with the name and the predictions passed in
// the passengers data in the following format:
// PassengerId,Survived
// 889,1
func (ptw PassengerTestWriter) Write(name string, predictions []float64) error {
	temp := "data/temp/"
	if _, err := os.Stat(temp); os.IsNotExist(err) {
		if err = os.Mkdir(temp, 0777); err != nil {
			log.Fatalln(err)
		}
	}

	csvfile, err := os.Create(temp + name)
	defer csvfile.Close()

	if err != nil {
		log.Fatalln(err)
	}

	writer := csv.NewWriter(csvfile)

	if err := writer.Write([]string{"PassengerId", "Survived"}); err != nil {
		log.Fatalln(err)
	}

	for i, passenger := range ptw.passengers {
		p := []string{passenger.ID, "0"}
		if predictions[i] == 1 {
			p[1] = "1"
		}
		if err := writer.Write(p); err != nil {
			log.Fatalln(err)
		}
	}
	writer.Flush()
	return nil
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
