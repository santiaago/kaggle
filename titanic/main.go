package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/santiaago/caltechx.go/linear"
	"github.com/santiaago/caltechx.go/linreg"
)

var (
	test  = flag.String("test", "data/test.csv", "testing set")
	train = flag.String("train", "data/train.csv", "training set")
)

func main() {
	flag.Parse()

	var linreg *linreg.LinearRegression
	// train
	if csvfile, err := os.Open(*train); err != nil {
		log.Fatalln(err)
	} else {
		reader := csv.NewReader(csvfile)
		linreg = trainModel(reader)
	}
	// test ...
	if csvfile, err := os.Open(*test); err != nil {
		log.Fatalln(err)
	} else {
		reader := csv.NewReader(csvfile)
		testModel(reader, linreg)
	}
}

func trainModel(r *csv.Reader) *linreg.LinearRegression {
	var rawData [][]string
	var err error
	if rawData, err = r.ReadAll(); err != nil {
		log.Fatalln(err)
		return nil
	}
	passengers := []passenger{}
	for i := 1; i < len(rawData); i++ {
		p := passengerFromTrainLine(rawData[i])
		passengers = append(passengers, p)
	}
	return linregSexAge(passengers)
	//linregPClassAge(passengers)
	//linregPClassSex(passengers)

}

func testModel(r *csv.Reader, linreg *linreg.LinearRegression) {
	var rawData [][]string
	var err error
	if rawData, err = r.ReadAll(); err != nil {
		log.Fatalln(err)
	}
	passengers := []passenger{}
	for i := 1; i < len(rawData); i++ {
		p := passengerFromTestLine(rawData[i])
		passengers = append(passengers, p)
	}
	linregTestSexAge(linreg, &passengers)
}

func linregTestSexAge(linreg *linreg.LinearRegression, passengers *[]passenger) {
	var data [][]float64
	for i := 0; i < len(*passengers); i++ {
		p := (*passengers)[i]

		var sex float64
		if p.Sex == "female" {
			sex = float64(1)
		}
		d := []float64{sex, float64(p.Age), float64(0)}
		data = append(data, d)
	}

	for i := 0; i < len(data); i++ {
		oX := data[i][:1]
		gi := float64(0)
		for j := 0; j < len(oX); j++ {
			gi += oX[j] * linreg.Wn[j]
		}

		if linear.Sign(gi) > 0 {
			data[i][2] = 1
			(*passengers)[i].Survived = true
		}
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

func linregPClassSex(passengers []passenger) *linreg.LinearRegression {
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

		var sex float64
		if p.Sex == "female" {
			sex = float64(1)
		}
		d := []float64{sex, pclass, survived}
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

func passengerFromTestLine(line []string) passenger {
	id := line[0]
	pclass := line[1]
	name := line[2]
	sex := line[3]
	age, err := strconv.ParseInt(line[4], 10, 32)
	if err != nil {
		age = 33
	}
	sibsp := line[5]
	parch := line[6]
	ticket := line[7]
	fare := line[8]
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
		sibsp,
		parch,
		ticket,
		fare,
		cabin,
		embarked,
	}
	return p
}
