// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"encoding/binary"
	"io"

	"github.com/nlpodyssey/safetensors/dtype"
)

// A Tensor with data fully loaded in memory.
// Data is interpreted and converted to a convenient type.
//
// For a correctly formed Tensor, the value of DType and the type of Data
// must match each other, according to the following pairs:
//
//	DType | Data type
//	------+---------------
//	Bool  | []bool
//	U8    | []uint8
//	I8    | []int8
//	U16   | []uint16
//	I16   | []int16
//	F16   | []float16.F16
//	BF16  | []float16.BF16
//	U32   | []uint32
//	I32   | []int32
//	F32   | []float32
//	U64   | []uint64
//	I64   | []int64
//	F64   | []float64
type Tensor struct {
	name  string
	dType dtype.DType
	shape []int
	data  any
}

// The Name of the tensor.
func (t Tensor) Name() string {
	return t.name
}

// DType returns the data type of the tensor.
func (t Tensor) DType() dtype.DType {
	return t.dType
}

// The Shape of the tensor. It can be nil.
func (t Tensor) Shape() []int {
	return t.shape
}

// The Data of the tensor.
// Possible values are documented on the main Tensor type.
func (t Tensor) Data() any {
	return t.data
}

// WriteTo converts the tensor's Data to safetensors-compliant byte format,
// writing the result to w.
// It satisfies io.WriterTo interface.
func (t Tensor) WriteTo(w io.Writer) (int64, error) {
	wc := writeCounter{w: w, counter: 0}
	// FIXME: binary.Write allocates too much
	err := binary.Write(&wc, binary.LittleEndian, t.data)
	return wc.counter, err
}

type writeCounter struct {
	w       io.Writer
	counter int64
}

func (wc *writeCounter) Write(p []byte) (int, error) {
	n, err := wc.w.Write(p)
	wc.counter += int64(n)
	return n, err
}
