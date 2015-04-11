//Package data reads, extracts and cleans data.
package data

import "fmt"

// A Reader reads data by extracting and cleaning the data.
type Reader struct {
	Ex Extractor // extracts the data
	Cl Cleaner   // cleans the data
}

// Read reads the data, by extracting it using the Extractor.Extract function,
// then performs a Cleaner.Clean and returns the cleaned data.
func (r Reader) Read() ([][]float64, error) {
	d, err := r.Ex.Extract()
	if err != nil {
		return nil, fmt.Errorf("error when extracting data: %v", err)
	}
	return r.Cl.Clean(d)
}

// Extractor is the interface that wraps the basic Extract method.
// Extract extracts the data from any source. It usually is in the form or []T
type Extractor interface {
	Extract() (interface{}, error)
}

// Cleaner is the interface that wraps the basic Clean method.
// Clean cleans the data passed in and return a two dimentional array of float64
// representing the data passed in.
type Cleaner interface {
	Clean(data interface{}) ([][]float64, error)
}
