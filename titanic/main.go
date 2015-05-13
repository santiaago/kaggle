package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"

	"github.com/santiaago/ml"
	"github.com/santiaago/ml/data"
)

var (
	test  = flag.String("test", "data/test.csv", "testing set")
	train = flag.String("train", "data/train.csv", "training set")

	importPath   = flag.String("ipath", "models.json", "path to a json array with models to use description.")
	importModels = flag.Bool("i", false, "defines if the program should import the models defined in ipath")

	exportPath   = flag.String("epath", "usedModels.json", "json array with the description of the trained models.")
	exportModels = flag.Bool("e", false, "defines if the program should export the used models defined in epath")

	tempPath = flag.String("temp", "data/temp/", "path of temp folder where all model results and rankings will be written.")

	trainLinreg = flag.Bool("linreg", false, "train linear regressions.")
	trainLogreg = flag.Bool("logreg", false, "train logistic regressions.")

	trainSpecific      = flag.Bool("specific", false, "train specific models.")
	combinations       = flag.Int("comb", 0, "number of features to try with all combinations.")
	trainTransforms    = flag.Bool("trans", false, "train models with transformations.")
	transformDimension = flag.Int("dim", 0, "dimension of transformation.")
	trainRegularized   = flag.Bool("reg", false, "train models with regularization.")

	einRank = flag.Bool("einRank", false, "writes a ranking.ein.md file with the in sample ranking of all processed models.")
	ecvRank = flag.Bool("ecvRank", false, "writes a ranking.ecv.md file with the cross validation ranking of all processed models.")
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

	if len(models) == 0 {
		fmt.Println("no models found.")
		return
	}

	var xTest data.Extractor
	var wTest data.Writer
	var drTest data.Reader

	xTest = NewPassengerTestExtractor()
	wTest = NewPassengerTestWriter(*test)
	drTest = NewPassengerReader(*test, xTest)

	testModels(drTest, wTest, models)

	rank(models)

	if *exportModels {
		// export models here..
		export(models, *exportPath)
	}

}

func export(models ml.ModelContainers, path string) {
	var a []modelInfo

	for m := range models {
		if models[m] == nil {
			continue
		}

		mi := ModelInfoFromModel(models[m])
		a = append(a, mi)
	}
	var b []byte
	var err error

	if b, err = json.Marshal(a); err != nil {
		log.Printf("unable to marshal array of modelInfo objects ", err)
	}

	err = ioutil.WriteFile(path, b, 0644)
	if err != nil {
		log.Printf("unable to write to file %v, %v", path, err)
		panic(err)
	}
}

func rank(models ml.ModelContainers) {
	fmt.Println("Start ranking models")
	if *einRank {
		fmt.Println("Start ranking models by Ein")
		writeEinRanking(models, "ranking.ein.md")
		models.TopEin(25)
		fmt.Println("Done ranking models by Ein")
	}

	if *ecvRank {
		fmt.Println("Start ranking models by Ecv")
		writeEcvRanking(models, "ranking.ecv.md")
		models.TopEcv(25)
		fmt.Println("Done ranking models by Ecv")
	}
	fmt.Println("Done ranking models")
}
