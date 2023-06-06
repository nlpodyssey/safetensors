// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package header

import (
	"fmt"
	"math"
	"math/bits"
	"sort"
)

// Validate checks whether the content of a Header is valid according to
// safetensors format, returning an error if a problem is encountered,
// otherwise nil.
//
// This validation can serve as an early isolated checking mechanism to
// identify bogus values before performing further actions that
// heavily depend upon the Header, such as reading tensors data from
// byte-buffer.
//
// The Header is checked against the following rules:
//
//   - ByteBufferOffset must not be negative
//   - each key in Tensors TensorMap must match the mapped Tensor.Name
//   - the union of DataOffsets of all Tensors must cover an entire contiguous
//     area of the byte-buffer, starting from offset 0
//   - DataOffsets of any pair of tensors must not overlap
//   - for each Tensor, its DataOffsets.Begin must be <= DataOffsets.End
//   - each Tensor's Shape must not contain negative values
//   - for each Tensor, its explicit byte size described by DataOffsets
//     (End - Begin) must coincide with the implicit byte size computed
//     from Shape and DType (product of all Shape items * DType size; an empty
//     shape counts as 1 scalar value)
//   - no overflow must occur during calculations at any step, making sure
//     that all computed values fit within the "int" type
func (h Header) Validate() error {
	if h.ByteBufferOffset < 0 {
		return fmt.Errorf("invalid byte-buffer offset negative value %d", h.ByteBufferOffset)
	}
	return validateTensors(h.Tensors)
}

func validateTensors(tm TensorMap) error {
	if err := validateTensorNames(tm); err != nil {
		return err
	}

	ts := tm.TensorSlice()
	sort.Sort(TensorSliceByDataOffsets{ts})

	expectedBegin := 0
	for _, t := range ts {
		if err := validateTensor(t, expectedBegin); err != nil {
			return fmt.Errorf("invalid tensor %q: %w", t.Name, err)
		}
		expectedBegin = t.DataOffsets.End
	}
	return nil
}

func validateTensorNames(tm TensorMap) error {
	for k, t := range tm {
		if k != t.Name {
			return fmt.Errorf("tensor names mismatch: TensorMap key %q, Tensor.Name %q", k, t.Name)
		}
	}
	return nil
}

func validateTensor(t Tensor, expectedBegin int) error {
	if t.DataOffsets.Begin != expectedBegin {
		return fmt.Errorf("expected data-offsets begin %d, actual %d", expectedBegin, t.DataOffsets.Begin)
	}
	if t.DataOffsets.End < t.DataOffsets.Begin {
		return fmt.Errorf("expected data-offsets end >= %d (begin), actual %d", t.DataOffsets.Begin, t.DataOffsets.End)
	}

	byteSize, err := byteSizeFromShape(t)
	if err != nil {
		return err
	}
	if offSize := t.DataOffsets.End - t.DataOffsets.Begin; offSize != byteSize {
		return fmt.Errorf("byte size computed from shape (%d) differs from data-offsets size (%d)", byteSize, offSize)
	}
	return nil
}

func byteSizeFromShape(t Tensor) (int, error) {
	if err := t.DType.Validate(); err != nil {
		return 0, err
	}

	tensorSize, err := tensorSizeFromShape(t.Shape)
	if err != nil {
		return 0, err
	}

	hi, byteSize := bits.Mul(tensorSize, uint(t.DType.Size()))
	if hi != 0 {
		return 0, fmt.Errorf("int overflow computing tensor byte size from shape")
	}
	if byteSize > math.MaxInt {
		return 0, fmt.Errorf("tensor byte size computed from shape is too large for int type: %d", byteSize)
	}
	return int(byteSize), nil
}

func tensorSizeFromShape(s Shape) (uint, error) {
	size := uint(1)
	for _, v := range s {
		if v < 0 {
			return 0, fmt.Errorf("shape contains negative value %d", v)
		}
		var hi uint
		if hi, size = bits.Mul(size, uint(v)); hi != 0 {
			return 0, fmt.Errorf("int overflow computing tensor elements size from shape")
		}
	}
	return size, nil
}
