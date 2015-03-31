package main

import "flag"

var (
	test  = flag.String("test", "data/test.csv", "testing set")
	train = flag.String("train", "data/train.csv", "training set")
)

func main() {
	flag.Parse()
	linregs, usedFeaturesPerModel := trainModels(*train)

	mapUsedFeatures := make(map[string][]int)
	for i := 0; i < len(linregs); i++ {
		mapUsedFeatures[linregs[i].Name] = usedFeaturesPerModel[i]
	}
	testModels(*test, linregs, mapUsedFeatures)
}
