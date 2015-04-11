package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/santiaago/kaggle/data"
)

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
func (pr PassengerReader) Read() ([][]float64, error) {
	d, err := pr.ex.Extract(pr.r)
	if err != nil {
		return nil, fmt.Errorf("error when extracting data: %v", err)
	}
	return pr.Clean(d)
}

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
