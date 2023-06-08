// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"bufio"
	"fmt"
	"io"
	"math"

	"github.com/nlpodyssey/safetensors/dtype"
	"github.com/nlpodyssey/safetensors/float16"
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

// NewTensor performs validity checks over the given properties and returns
// a Tensor with those properties if validation succeeds, otherwise an error.
//
// If the error returned is not nil, the Tensor is a zero-value that
// must not be used.
//
// Here is an overview of the rules applied for validation:
//   - an empty name ("") is allowed
//   - the dType must be valid (see dtype.DType.Validate)
//   - an empty or nil shape is allowed (a scalar value is implied)
//   - the shape must not contain negative values
//   - the type of data must match the dType, according to the pairs listed
//     on Tensor documentation
//   - the number of data elements must match the shape
//
// These rules are in place as a minimum guarantee that the Tensor can be
// serialized correctly on a later stage. For the same reason, the given
// shape is copied before being assigned to the Tensor.
//
// Since "data" can possibly take a large amount of memory, its value is NOT
// copied, and is directly assigned to the Tensor. Accidental modifications
// to the data given to this function could lead to subsequent unexpected
// content or corrupted serialization, even in absence of errors.
func NewTensor(name string, dType dtype.DType, shape []int, data any) (Tensor, error) {
	dataLen, err := checkTypesAndGetDataLen(dType, data)
	if err != nil {
		return Tensor{}, err
	}
	shapeSize, err := checkedShapeSize(shape)
	if err != nil {
		return Tensor{}, err
	}
	if shapeSize != dataLen {
		return Tensor{}, fmt.Errorf("the size computed from shape (%d) does not match data length (%d)", shapeSize, dataLen)
	}
	return Tensor{
		name:  name,
		dType: dType,
		shape: copyShape(shape),
		data:  data,
	}, nil
}

func checkedShapeSize(shape []int) (int, error) {
	size := 1
	for _, v := range shape {
		if v < 0 {
			return 0, fmt.Errorf("shape contains a negative value")
		}
		size *= v
	}
	return size, nil
}

func checkTypesAndGetDataLen(dt dtype.DType, data any) (int, error) {
	switch dt {
	case dtype.Bool:
		return resolveDataLen[bool](dt, data)
	case dtype.U8:
		return resolveDataLen[uint8](dt, data)
	case dtype.I8:
		return resolveDataLen[int8](dt, data)
	case dtype.U16:
		return resolveDataLen[uint16](dt, data)
	case dtype.I16:
		return resolveDataLen[int16](dt, data)
	case dtype.F16:
		return resolveDataLen[float16.F16](dt, data)
	case dtype.BF16:
		return resolveDataLen[float16.BF16](dt, data)
	case dtype.U32:
		return resolveDataLen[uint32](dt, data)
	case dtype.I32:
		return resolveDataLen[int32](dt, data)
	case dtype.F32:
		return resolveDataLen[float32](dt, data)
	case dtype.U64:
		return resolveDataLen[uint64](dt, data)
	case dtype.I64:
		return resolveDataLen[int64](dt, data)
	case dtype.F64:
		return resolveDataLen[float64](dt, data)
	}
	return 0, fmt.Errorf("invalid or unsupported DType: %s", dt)
}

func resolveDataLen[T any](dt dtype.DType, data any) (int, error) {
	if data == nil {
		return 0, nil
	}
	y, ok := data.([]T)
	if !ok {
		return 0, fmt.Errorf("expected DType %s to match data type %T, actual data type %T", dt, y, data)
	}
	return len(y), nil
}

// The Name of the tensor.
func (t Tensor) Name() string {
	return t.name
}

// DType returns the data type of the tensor.
func (t Tensor) DType() dtype.DType {
	return t.dType
}

// The Shape of the tensor.
//
// If the shape is zero-length, it returns nil, otherwise a new slice
// is allocated and returned (the shape is copied to prevent tampering).
func (t Tensor) Shape() []int {
	return copyShape(t.shape)
}

// The Data of the tensor.
// Possible values are documented on the main Tensor type.
//
// The value returned is NOT a copy: any change to its content will
// affect the Tensor too. Accidental modifications could lead to subsequent
// unexpected content or corrupted serialization, even in absence of errors.
func (t Tensor) Data() any {
	return t.data
}

// WriteTo converts the tensor's Data to safetensors-compliant byte format,
// writing the result to w.
// It satisfies io.WriterTo interface.
func (t Tensor) WriteTo(w io.Writer) (int64, error) {
	bw := bufio.NewWriter(w)
	n, err := t.writeTo(bw)
	if e := bw.Flush(); e != nil && err == nil {
		err = e
	}
	return n, err
}

func (t Tensor) writeTo(w io.Writer) (int64, error) {
	switch t.dType {
	case dtype.Bool:
		return writeBoolData(w, t.data)
	case dtype.U8:
		return writeU8Data(w, t.data)
	case dtype.I8:
		return writeI8Data(w, t.data)
	case dtype.U16:
		return write16bitData[uint16](w, t.data)
	case dtype.I16:
		return write16bitData[int16](w, t.data)
	case dtype.F16:
		return write16bitData[float16.F16](w, t.data)
	case dtype.BF16:
		return write16bitData[float16.BF16](w, t.data)
	case dtype.U32:
		return write32bitData[uint32](w, t.data)
	case dtype.I32:
		return write32bitData[int32](w, t.data)
	case dtype.F32:
		return writeF32Data(w, t.data)
	case dtype.U64:
		return write64bitData[uint64](w, t.data)
	case dtype.I64:
		return write64bitData[int64](w, t.data)
	case dtype.F64:
		return writeF64Data(w, t.data)
	}
	return 0, fmt.Errorf("invalid or unsupported DType: %s", t.dType)
}

func writeBoolData(w io.Writer, data any) (int64, error) {
	v, err := castSlice[bool](data)
	if err != nil {
		return 0, err
	}

	a := [2]byte{0, 1}
	zero := a[0:1]
	one := a[1:2]

	written := 0
	for _, x := range v {
		var n int
		if x {
			n, err = w.Write(one)
		} else {
			n, err = w.Write(zero)
		}
		written += n
		if err != nil {
			return int64(written), err
		}
	}
	return int64(written), nil
}

func writeU8Data(w io.Writer, data any) (int64, error) {
	v, err := castSlice[uint8](data)
	if err != nil {
		return 0, err
	}

	written := 0
	for i := range v {
		n, err := w.Write(v[i : i+1])
		written += n
		if err != nil {
			return int64(written), err
		}
	}
	return int64(written), nil
}

func writeI8Data(w io.Writer, data any) (int64, error) {
	v, err := castSlice[int8](data)
	if err != nil {
		return 0, err
	}

	var a [1]byte
	b := a[:]

	written := 0
	for _, x := range v {
		a[0] = byte(x)
		n, err := w.Write(b)
		written += n
		if err != nil {
			return int64(written), err
		}
	}
	return int64(written), nil
}

func write16bitData[T uint16 | int16 | float16.F16 | float16.BF16](w io.Writer, data any) (int64, error) {
	v, err := castSlice[T](data)
	if err != nil {
		return 0, err
	}

	var a [2]byte
	b := a[:]

	written := 0
	for _, x := range v {
		a[0] = byte(x)
		a[1] = byte(x >> 8)

		n, err := w.Write(b)
		written += n
		if err != nil {
			return int64(written), err
		}
	}
	return int64(written), nil
}

func write32bitData[T uint32 | int32](w io.Writer, data any) (int64, error) {
	v, err := castSlice[T](data)
	if err != nil {
		return 0, err
	}

	var a [4]byte
	b := a[:]

	written := 0
	for _, x := range v {
		a[0] = byte(x)
		a[1] = byte(x >> 8)
		a[2] = byte(x >> 16)
		a[3] = byte(x >> 24)

		n, err := w.Write(b)
		written += n
		if err != nil {
			return int64(written), err
		}
	}
	return int64(written), nil
}

func writeF32Data(w io.Writer, data any) (int64, error) {
	v, err := castSlice[float32](data)
	if err != nil {
		return 0, err
	}

	var a [4]byte
	b := a[:]

	written := 0
	for _, x := range v {
		u := math.Float32bits(x)
		a[0] = byte(u)
		a[1] = byte(u >> 8)
		a[2] = byte(u >> 16)
		a[3] = byte(u >> 24)

		n, err := w.Write(b)
		written += n
		if err != nil {
			return int64(written), err
		}
	}
	return int64(written), nil
}

func write64bitData[T uint64 | int64](w io.Writer, data any) (int64, error) {
	v, err := castSlice[T](data)
	if err != nil {
		return 0, err
	}

	var a [8]byte
	b := a[:]

	written := 0
	for _, x := range v {
		a[0] = byte(x)
		a[1] = byte(x >> 8)
		a[2] = byte(x >> 16)
		a[3] = byte(x >> 24)
		a[4] = byte(x >> 32)
		a[5] = byte(x >> 40)
		a[6] = byte(x >> 48)
		a[7] = byte(x >> 56)

		n, err := w.Write(b)
		written += n
		if err != nil {
			return int64(written), err
		}
	}
	return int64(written), nil
}

func writeF64Data(w io.Writer, data any) (int64, error) {
	v, err := castSlice[float64](data)
	if err != nil {
		return 0, err
	}

	var a [8]byte
	b := a[:]

	written := 0
	for _, x := range v {
		u := math.Float64bits(x)
		a[0] = byte(u)
		a[1] = byte(u >> 8)
		a[2] = byte(u >> 16)
		a[3] = byte(u >> 24)
		a[4] = byte(u >> 32)
		a[5] = byte(u >> 40)
		a[6] = byte(u >> 48)
		a[7] = byte(u >> 56)

		n, err := w.Write(b)
		written += n
		if err != nil {
			return int64(written), err
		}
	}
	return int64(written), nil
}

func castSlice[T any](x any) ([]T, error) {
	y, ok := x.([]T)
	if !ok {
		return y, fmt.Errorf("expected data type %T, actual %T", x, y)
	}
	return y, nil
}
