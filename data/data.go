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
	Write(filename string, predictions []float64) error
}

// Container holds the data and features information.
type Container struct {
	Data     [][]float64 // array that holds the information to learn or train.
	Features []int       // array of indexes with the features to use in the Data array.
	Predict  int         // index of the data array that tells how to classify.
}

// Filter returns a 2D array of floats filtered with the params passed in.
// It appends the defined Predict column 'Y' coordinate at the end of each row.
// Like so:
// x1 x2 x3
// x1 x2 x3
func (c Container) Filter(keep []int) (filtered [][]float64) {

	for i := 0; i < len(c.Data); i++ {
		var row []float64
		for j := 0; j < len(keep); j++ {
			row = append(row, c.Data[i][keep[j]])
		}
		filtered = append(filtered, row)
	}
	return
}

// Filter returns a 2D array of floats filtered with the params passed in.
// It appends the defined Predict column 'Y' coordinate at the end of each row.
// Like so:
// x1 x2 x3 y
// x1 x2 x3 y
func (c Container) FilterWithPredict(keep []int) (filtered [][]float64) {

	for i := 0; i < len(c.Data); i++ {
		var row []float64
		for j := 0; j < len(keep); j++ {
			row = append(row, c.Data[i][keep[j]])
		}

		row = append(row, c.Data[i][c.Predict])
		filtered = append(filtered, row)
	}
	return
}
