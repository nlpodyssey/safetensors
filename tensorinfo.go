// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

// TensorInfo provides information of a single tensor.
// Endianness is assumed to be little-endian. Ordering is assumed to be 'C'.
type TensorInfo struct {
	// The DType of each element of the tensor.
	DType DType `json:"dtype"`
	// The Shape of the tensor.
	Shape []uint64 `json:"shape"`
	// DataOffsets provides the offsets to find the data
	// within the byte-buffer array.
	DataOffsets [2]uint64 `json:"data_offsets"`
}

// NamedTensorInfo is a pair of a TensorInfo and its name (or label, or key).
type NamedTensorInfo struct {
	Name       string
	TensorInfo TensorInfo
}
