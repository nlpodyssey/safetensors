// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import "fmt"

// TensorView is a view of a Tensor within a file.
//
// It contains references to data within the full byte-buffer
// and is thus a readable view of a single tensor.
type TensorView struct {
	dType DType
	shape []uint64
	data  []byte
}

// NamedTensorView is a pair of a TensorView and its name (or label, or key).
type NamedTensorView struct {
	Name       string
	TensorView TensorView
}

func (tv TensorView) DType() DType    { return tv.dType }
func (tv TensorView) Shape() []uint64 { return tv.shape }
func (tv TensorView) Data() []byte    { return tv.data }
func (tv TensorView) DataLen() uint64 { return uint64(len(tv.data)) }

// NewTensorView creates a new TensorView.
func NewTensorView(dType DType, shape []uint64, data []byte) (TensorView, error) {
	n := uint64(len(data))
	numElements := numElementsFromShape(shape)

	if n != numElements*dType.Size() {
		return TensorView{}, fmt.Errorf("invalid tensor view: dtype=%d shape=%+v len(data)=%d", dType, shape, n)
	}

	return TensorView{
		dType: dType,
		shape: shape,
		data:  data,
	}, nil
}

func numElementsFromShape(shape []uint64) uint64 {
	if len(shape) == 0 {
		return 0
	}
	n := shape[0]
	for _, v := range shape[1:] {
		n *= v
	}
	return n
}
