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
	return 0, fmt.Errorf("invalid or unsupported DType %s", t.dType)
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
