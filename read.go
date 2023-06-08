// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"bufio"
	"fmt"
	"io"
	"math"
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

func copyShape(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}
	s := make([]int, len(shape))
	copy(s, shape)
	return s
}

func readTypedTensorData(ht header.Tensor, r io.Reader) (any, error) {
	byteSize := ht.DataOffsets.End - ht.DataOffsets.Begin
	typedSize := byteSize / ht.DType.Size()
	lr := io.LimitedReader{R: r, N: int64(byteSize)}
	br := bufio.NewReader(&lr)

	switch ht.DType {
	case dtype.Bool:
		return readBoolData(br, typedSize)
	case dtype.U8:
		return readU8Data(br, typedSize)
	case dtype.I8:
		return readI8Data(br, typedSize)
	case dtype.U16:
		return read16bitData[uint16](br, typedSize)
	case dtype.I16:
		return read16bitData[int16](br, typedSize)
	case dtype.F16:
		return read16bitData[float16.F16](br, typedSize)
	case dtype.BF16:
		return read16bitData[float16.BF16](br, typedSize)
	case dtype.U32:
		return read32bitData[uint32](br, typedSize)
	case dtype.I32:
		return read32bitData[int32](br, typedSize)
	case dtype.F32:
		return readF32Data(br, typedSize)
	case dtype.U64:
		return read64bitData[uint64](br, typedSize)
	case dtype.I64:
		return read64bitData[int64](br, typedSize)
	case dtype.F64:
		return readF64Data(br, typedSize)
	}
	return nil, fmt.Errorf("invalid or unsupported DType %s", ht.DType)
}

func readBoolData(r io.Reader, size int) ([]bool, error) {
	var a [1]byte
	b := a[:]

	out := make([]bool, size)
	for i := range out {
		if _, err := io.ReadFull(r, b); err != nil {
			return nil, err
		}
		out[i] = a[0] != 0
	}
	return out, nil
}

func readU8Data(r io.Reader, size int) ([]uint8, error) {
	out := make([]uint8, size)
	for i := range out {
		if _, err := io.ReadFull(r, out[i:i+1]); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func readI8Data(r io.Reader, size int) ([]int8, error) {
	var a [1]byte
	b := a[:]

	out := make([]int8, size)
	for i := range out {
		if _, err := io.ReadFull(r, b); err != nil {
			return nil, err
		}
		out[i] = int8(a[0])
	}
	return out, nil
}

func read16bitData[T uint16 | int16 | float16.F16 | float16.BF16](r io.Reader, size int) ([]T, error) {
	var a [2]byte
	b := a[:]

	out := make([]T, size)
	for i := range out {
		if _, err := io.ReadFull(r, b); err != nil {
			return nil, err
		}
		out[i] = T(a[0]) | T(a[1])<<8
	}
	return out, nil
}

func read32bitData[T uint32 | int32](r io.Reader, size int) ([]T, error) {
	var a [4]byte
	b := a[:]

	out := make([]T, size)
	for i := range out {
		if _, err := io.ReadFull(r, b); err != nil {
			return nil, err
		}
		out[i] = T(a[0]) | T(a[1])<<8 | T(a[2])<<16 | T(a[3])<<24
	}
	return out, nil
}

func readF32Data(r io.Reader, size int) ([]float32, error) {
	var a [4]byte
	b := a[:]

	out := make([]float32, size)
	for i := range out {
		if _, err := io.ReadFull(r, b); err != nil {
			return nil, err
		}
		out[i] = math.Float32frombits(
			uint32(a[0]) | uint32(a[1])<<8 | uint32(a[2])<<16 | uint32(a[3])<<24)
	}
	return out, nil
}

func read64bitData[T uint64 | int64](r io.Reader, size int) ([]T, error) {
	var a [8]byte
	b := a[:]

	out := make([]T, size)
	for i := range out {
		if _, err := io.ReadFull(r, b); err != nil {
			return nil, err
		}
		out[i] = T(b[0]) | T(b[1])<<8 | T(b[2])<<16 | T(b[3])<<24 |
			T(b[4])<<32 | T(b[5])<<40 | T(b[6])<<48 | T(b[7])<<56
	}
	return out, nil
}

func readF64Data(r io.Reader, size int) ([]float64, error) {
	var a [8]byte
	b := a[:]

	out := make([]float64, size)
	for i := range out {
		if _, err := io.ReadFull(r, b); err != nil {
			return nil, err
		}
		out[i] = math.Float64frombits(
			uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
				uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56)
	}
	return out, nil
}
