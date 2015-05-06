package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"

	"github.com/santiaago/ml"
	"github.com/santiaago/ml/data"
)

var (
	test  = flag.String("test", "data/test.csv", "testing set")
	train = flag.String("train", "data/train.csv", "training set")
)

func init() {
	log.SetFlags(log.Ltime | log.Ldate | log.Lshortfile)
}

func main() {
	flag.Parse()

	var xTrain data.Extractor
	var drTrain data.Reader
	xTrain = NewPassengerTrainExtractor()
	drTrain = NewPassengerReader(*train, xTrain)

	models := trainModels(drTrain)

	var xTest data.Extractor
	var wTest data.Writer
	var drTest data.Reader

	xTest = NewPassengerTestExtractor()
	wTest = NewPassengerTestWriter(*test)
	drTest = NewPassengerReader(*test, xTest)

	if len(models) == 0 {
		fmt.Println("no models found.")
		return
	}

	testModels(drTest, wTest, models)

	// writeRankingEin(models, "ranking.ein.md")
	// todo(santiaago): too slow
	// writeRankingEcv(models, "ranking.ecv.md")

	models.PrintTop(50)
}

func writeRankingEin(models ml.ModelContainers, name string) {
	temp := "data/temp/"
	if _, err := os.Stat(temp); os.IsNotExist(err) {
		if err = os.Mkdir(temp, 0777); err != nil {
			log.Fatalln(err)
		}
	}

	file, err := os.Create(temp + name)
	defer file.Close()

	if err != nil {
		log.Fatalln(err)
	}

	writer := bufio.NewWriter(file)

	if _, err := writer.WriteString("model ranking\n"); err != nil {
		log.Fatalln(err)
	}

	sort.Sort(ml.ByEin(models))
	for i, m := range models {
		if m == nil || m.Model == nil {
			continue
		}

		line := fmt.Sprintf("%v\t\tEin = %f\tmodel: %v\n", i, m.Model.Ein(), m.Name)
		if _, err := writer.WriteString(line); err != nil {
			log.Fatalln(err)
		}
		writer.Flush()
	}
	writer.Flush()
}

func writeRankingEcv(models ml.ModelContainers, name string) {
	temp := "data/temp/"
	if _, err := os.Stat(temp); os.IsNotExist(err) {
		if err = os.Mkdir(temp, 0777); err != nil {
			log.Fatalln(err)
		}
	}

	file, err := os.Create(temp + name)
	defer file.Close()

	if err != nil {
		log.Fatalln(err)
	}

	writer := bufio.NewWriter(file)

	if _, err := writer.WriteString("model ranking\n"); err != nil {
		log.Fatalln(err)
	}

	sort.Sort(ml.ByEcv(models))
	for i, m := range models {
		if m == nil || m.Model == nil {
			continue
		}

		line := fmt.Sprintf("%v\t\tEcv = %f\tmodel: %v\n", i, m.Model.Ecv(), m.Name)
		if _, err := writer.WriteString(line); err != nil {
			log.Fatalln(err)
		}
		writer.Flush()
	}
	writer.Flush()
}
