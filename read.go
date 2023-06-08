// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"encoding/binary"
	"fmt"
	"io"
	"sort"

	"github.com/nlpodyssey/safetensors/dtype"
	"github.com/nlpodyssey/safetensors/float16"
	"github.com/nlpodyssey/safetensors/header"
)

// ST (short for "SafeTensors") is the result of reading the full content
// of a safetensors data stream (or file), loading full Tensor objects in
// memory.
type ST struct {
	Tensors  []Tensor
	Metadata map[string]string
}

// RawST (short for "SafeTensors") is the result of reading the full content
// of a safetensors data stream (or file), loading full RawTensor objects in
// memory.
type RawST struct {
	Tensors  []RawTensor
	Metadata map[string]string
}

// ReadAll reads and interprets the whole content of a safetensors data
// stream (or file). After reading successfully the header part,
// the data of each tensor is read and loaded in memory, converted to
// typed data within Tensor objects.
//
// If headerSizeLimit is set to a positive number, its value is used to
// limit the reading of safetensors header. This can be useful to guard
// against attacks or tampered/garbage data, avoiding giant memory allocations
// to hold header information. A value of zero, or a negative number, have
// no limiting effects.
func ReadAll(r io.Reader, headerSizeLimit int) (ST, error) {
	head, err := readValidHeader(r, headerSizeLimit)
	if err != nil {
		return ST{}, err
	}

	tensors, err := readAllTensors(head.Tensors, r, false)
	if err != nil {
		return ST{}, err
	}

	return ST{
		Tensors:  tensors,
		Metadata: head.Metadata,
	}, nil
}

// ReadAllRaw is similar to ReadAll, but returns raw tensors data.
func ReadAllRaw(r io.Reader, headerSizeLimit int) (RawST, error) {
	head, err := readValidHeader(r, headerSizeLimit)
	if err != nil {
		return RawST{}, err
	}

	tensors, err := readAllRawTensors(head.Tensors, r, false)
	if err != nil {
		return RawST{}, err
	}

	return RawST{
		Tensors:  tensors,
		Metadata: head.Metadata,
	}, nil
}

func readValidHeader(r io.Reader, sizeLimit int) (header.Header, error) {
	if sizeLimit > 0 {
		r = io.LimitReader(r, int64(sizeLimit))
	}
	head, err := header.Read(r)
	if err != nil {
		return header.Header{}, fmt.Errorf("failed to read safetensors header: %w", err)
	}
	if err = head.Validate(); err != nil {
		return header.Header{}, fmt.Errorf("safetensors header is invalid: %w", err)
	}
	return head, nil
}

func readAllTensors(tm header.TensorMap, r io.Reader, safeCopy bool) ([]Tensor, error) {
	tensorSlice := tm.TensorSlice()
	sort.Sort(header.TensorSliceByDataOffsets{TensorSlice: tensorSlice})

	out := make([]Tensor, len(tensorSlice))
	for i, ht := range tensorSlice {
		var err error
		if out[i], err = readTensor(ht, r, safeCopy); err != nil {
			return nil, fmt.Errorf("failed to read data of tensor %q: %w", ht.Name, err)
		}
	}
	return out, nil
}

func readAllRawTensors(tm header.TensorMap, r io.Reader, safeCopy bool) ([]RawTensor, error) {
	tensorSlice := tm.TensorSlice()
	sort.Sort(header.TensorSliceByDataOffsets{TensorSlice: tensorSlice})

	out := make([]RawTensor, len(tensorSlice))
	for i, ht := range tensorSlice {
		var err error
		if out[i], err = readRawTensor(ht, r, safeCopy); err != nil {
			return nil, fmt.Errorf("failed to read data of tensor %q: %w", ht.Name, err)
		}
	}
	return out, nil
}

func readTensor(ht header.Tensor, r io.Reader, safeCopy bool) (Tensor, error) {
	data, err := readTypedTensorData(ht, r)
	if err != nil {
		return Tensor{}, err
	}
	shape := ht.Shape
	if safeCopy {
		shape = copyShape(shape)
	}
	return Tensor{
		name:  ht.Name,
		dType: ht.DType,
		shape: shape,
		data:  data,
	}, nil
}

func readRawTensor(ht header.Tensor, r io.Reader, safeCopy bool) (RawTensor, error) {
	shape := ht.Shape
	if safeCopy {
		shape = copyShape(shape)
	}
	rt := RawTensor{
		name:  ht.Name,
		dType: ht.DType,
		shape: shape,
		data:  nil,
	}

	size := ht.DataOffsets.End - ht.DataOffsets.Begin
	if size == 0 {
		return rt, nil
	}

	rt.data = make([]byte, size)
	if _, err := io.ReadFull(r, rt.data); err != nil {
		return RawTensor{}, fmt.Errorf("failed to read tensor data: %w", err)
	}
	return rt, nil
}

func readTypedTensorData(ht header.Tensor, r io.Reader) (any, error) {
	data, err := makeTypedTensorData(ht)
	if err != nil {
		return nil, err
	}
	// FIXME: binary.Read allocates too much
	if err = binary.Read(r, binary.LittleEndian, data); err != nil {
		return Tensor{}, fmt.Errorf("failed to read and convert tensor data: %w", err)
	}
	return data, nil
}

func makeTypedTensorData(ht header.Tensor) (any, error) {
	size := (ht.DataOffsets.End - ht.DataOffsets.Begin) / ht.DType.Size()
	switch ht.DType {
	case dtype.Bool:
		return make([]bool, size), nil
	case dtype.U8:
		return make([]uint8, size), nil
	case dtype.I8:
		return make([]int8, size), nil
	case dtype.U16:
		return make([]uint16, size), nil
	case dtype.I16:
		return make([]int16, size), nil
	case dtype.F16:
		return make([]float16.F16, size), nil
	case dtype.BF16:
		return make([]float16.BF16, size), nil
	case dtype.U32:
		return make([]uint32, size), nil
	case dtype.I32:
		return make([]int32, size), nil
	case dtype.F32:
		return make([]float32, size), nil
	case dtype.U64:
		return make([]uint64, size), nil
	case dtype.I64:
		return make([]int64, size), nil
	case dtype.F64:
		return make([]float64, size), nil
	}
	return nil, fmt.Errorf("invalid or unsupported DType %s", ht.DType)
}

func copyShape(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}
	s := make([]int, len(shape))
	copy(s, shape)
	return s
}
