// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"github.com/nlpodyssey/safetensors/dtype"
)

// RawTensor is a tensor with data fully loaded in memory.
//
// Unlike Tensor, Data is provided in its raw format, as it is read from
// a safetensors stream (or file), without being interpreted and converted
// to a specifically typed slice.
type RawTensor struct {
	name  string
	dType dtype.DType
	shape []int
	data  []byte
}

// The Name of the tensor.
func (rt RawTensor) Name() string {
	return rt.name
}

// DType returns the data type of the tensor.
func (rt RawTensor) DType() dtype.DType {
	return rt.dType
}

// The Shape of the tensor. It can be nil.
func (rt RawTensor) Shape() []int {
	return rt.shape
}

// Data returns the raw data of the tensor.
// It is expected to be little-endian and row-major ("C") ordered.
// There is no striding.
func (rt RawTensor) Data() []byte {
	return rt.data
}
