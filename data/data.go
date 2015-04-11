//Package data reads, extracts and cleans data.
package data

import "encoding/csv"

// Reader is the interface that wraps the basic read method.
// Read returns a 2 Dimensional array of floats.
// Use it to extract and clean data that you can then pass to a machine learning
// model.
type Reader interface {
	Read() (Container, error)
}

// Extractor is the interface that wraps the basic extract method.
// Extract returns anything and an error.
// Use it to extract data that you can then pass to a machine learning
// model.
type Extractor interface {
	Extract(r *csv.Reader) (interface{}, error)
}

type Writer interface {
	Write(filename string, predictions []int) error
}

// Container holds the data and features information.
type Container struct {
	Data     [][]float64 // array that holds the information to learn or train.
	Features []int       // array of indexes with the features to use in the Data array.
}
