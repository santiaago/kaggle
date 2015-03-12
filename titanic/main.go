package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"strings"
)

var (
	test  = flag.String("test", "data/test.csv", "testing set")
	train = flag.String("train", "data/train.csv", "training set")
)

func main() {
	flag.Parse()

	// train
	if data, err := ioutil.ReadFile(*train); err != nil {
		log.Fatalln(err)
	} else {
		reader := strings.NewReader(fmt.Sprintf("%s", data))
		trainModel(reader)
	}
	// test ...
}

func trainModel(r *strings.Reader) {
	fmt.Println("train")
}
