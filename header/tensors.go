// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package header

import (
	"github.com/nlpodyssey/safetensors/dtype"
)

// Tensor provides properties of a tensor, as described within a
// safetensors header.
type Tensor struct {
	Name        string
	DType       dtype.DType
	Shape       Shape
	DataOffsets DataOffsets
}

// TensorMap is a set of Tensor objects mapped by their name.
type TensorMap map[string]Tensor

// TensorSlice is a slice of Tensor objects.
type TensorSlice []Tensor

// TensorSliceByDataOffsets implements sort.Interface allowing to sort a
// TensorSlice by ascending DataOffsets values.
// It provides Less, while using Len and Swap methods of the embedded
// TensorSlice value.
type TensorSliceByDataOffsets struct{ TensorSlice }

// Less reports whether DataOffsets "a" is ordered before DataOffsets "b".
func (a DataOffsets) Less(b DataOffsets) bool {
	return a.Begin < b.Begin || (a.Begin == b.Begin && a.End < b.End)
}

// TensorSlice creates an unsorted slice of Tensor objects filled with
// all values of the TensorMap.
func (tm TensorMap) TensorSlice() TensorSlice {
	if len(tm) == 0 {
		return nil
	}
	ts := make(TensorSlice, 0, len(tm))
	for _, t := range tm {
		ts = append(ts, t)
	}
	return ts
}

// Len is the number of elements in the collection.
// This function partially satisfies sort.Interface.
func (ts TensorSlice) Len() int {
	return len(ts)
}

// Swap swaps the elements with indexes i and j.
// This function partially satisfies sort.Interface.
func (ts TensorSlice) Swap(i, j int) {
	ts[i], ts[j] = ts[j], ts[i]
}

// Less reports whether the Tensor with index i must sort before the Tensor
// with index j, according to their DataOffsets.
func (t TensorSliceByDataOffsets) Less(i, j int) bool {
	return t.TensorSlice[i].DataOffsets.Less(t.TensorSlice[j].DataOffsets)
}
