package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/santiaago/caltechx.go/linreg"
)

var (
	test  = flag.String("test", "data/test.csv", "testing set")
	train = flag.String("train", "data/train.csv", "training set")
)

func main() {
	flag.Parse()

	// train
	if csvfile, err := os.Open(*train); err != nil {
		log.Fatalln(err)
	} else {
		reader := csv.NewReader(csvfile)
		trainModel(reader)
	}
	// test ...
}

func trainModel(r *csv.Reader) {
	if rawData, err := r.ReadAll(); err != nil {
		log.Fatalln(err)
	} else {
		passengers := []passenger{}
		for i := 1; i < len(rawData); i++ {
			p := passengerFromTrainLine(rawData[i])
			passengers = append(passengers, p)
		}
		linregSexAge(passengers)
		linregPClassAge(passengers)

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

func passengerFromTrainLine(line []string) passenger {

	survived, err := strconv.ParseBool(line[1])
	if err != nil {
		survived = false
	}
	age, err := strconv.ParseInt(line[5], 10, 32)
	if err != nil {
		age = 33
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
		line[6],
		line[7],
		line[8],
		line[9],
		line[10],
		embarked,
	}
	return p
}
