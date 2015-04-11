package main

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"
)

type PassengerTestExtractor struct{}

func NewPassengerTestExtractor() PassengerTestExtractor {
	return PassengerTestExtractor{}
}

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

type PassengerTestWriter struct {
	passengers []passenger
	ex         PassengerTestExtractor
}

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

func (ptw PassengerTestWriter) Write(name string, predictions []int) error {
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
	// headers
	if err := writer.Write([]string{"PassengerId", "Survived"}); err != nil {
		log.Fatalln(err)
	}
	// data
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
