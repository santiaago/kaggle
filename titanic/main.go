package main

import "flag"

var (
	test  = flag.String("test", "data/test.csv", "testing set")
	train = flag.String("train", "data/train.csv", "training set")
)

func main() {
	flag.Parse()
	linregs, names := trainModels(*train)
	testModels(*test, linregs, names)
}
