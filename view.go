// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

// View is an interface to enable safetensors to serialize a tensor.
type View interface {
	// The DType of the tensor.
	DType() DType

	// The Shape of the tensor.
	Shape() []uint64

	// The Data of the tensor.
	Data() []byte

	// DataLen returns the length of the data in bytes.
	//
	// This is necessary as this might be faster to get than `len(Data())`.
	DataLen() uint64
}

// NamedView is a pair of a View and its name (or label, or key).
type NamedView[V View] struct {
	Name string
	View V
}
