// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package header

import (
	"math"
	"testing"

	"github.com/nlpodyssey/safetensors/dtype"
	"github.com/stretchr/testify/assert"
)

func TestHeader_Validate_Success(t *testing.T) {
	testCases := []struct {
		name string
		h    Header
	}{
		{"no tensors", Header{}},
		{"one tensor", Header{Tensors: TensorMap{
			"a": Tensor{
				Name:        "a",
				DType:       dtype.U8,
				Shape:       Shape{2, 3},
				DataOffsets: DataOffsets{0, 6},
			},
		}}},
		{"tensors of different types", Header{Tensors: TensorMap{
			"a": Tensor{
				Name:        "a",
				DType:       dtype.Bool,
				Shape:       Shape{2, 5},
				DataOffsets: DataOffsets{0, 10},
			},
			"b": Tensor{
				Name:        "b",
				DType:       dtype.U16,
				Shape:       Shape{5, 4},
				DataOffsets: DataOffsets{10, 50},
			},
			"c": Tensor{
				Name:        "c",
				DType:       dtype.F32,
				Shape:       Shape{15},
				DataOffsets: DataOffsets{50, 110},
			},
			"d": Tensor{
				Name:        "d",
				DType:       dtype.I64,
				Shape:       Shape{3, 5},
				DataOffsets: DataOffsets{110, 230},
			},
		}}},

		{"zero size", Header{Tensors: TensorMap{
			"a": Tensor{
				Name:        "a",
				DType:       dtype.U8,
				Shape:       Shape{0},
				DataOffsets: DataOffsets{0, 0},
			},
			"b": Tensor{
				Name:        "b",
				DType:       dtype.U16,
				Shape:       Shape{2, 0},
				DataOffsets: DataOffsets{0, 0},
			},
		}}},
		{"empty shapes", Header{Tensors: TensorMap{
			"a": Tensor{
				Name:        "a",
				DType:       dtype.U8,
				Shape:       nil,
				DataOffsets: DataOffsets{0, 1},
			},
			"b": Tensor{
				Name:        "b",
				DType:       dtype.U16,
				Shape:       nil,
				DataOffsets: DataOffsets{1, 3},
			},
			"c": Tensor{
				Name:        "c",
				DType:       dtype.U32,
				Shape:       Shape{},
				DataOffsets: DataOffsets{3, 7},
			},
			"d": Tensor{
				Name:        "d",
				DType:       dtype.U64,
				Shape:       Shape{},
				DataOffsets: DataOffsets{7, 15},
			},
		}}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.h.Validate()
			assert.NoError(t, err)
		})
	}
}

func TestHeader_Validate_Failure(t *testing.T) {
	testCases := []struct {
		name   string
		h      Header
		errMsg string
	}{
		{
			"negative ByteBufferOffset",
			Header{ByteBufferOffset: -1},
			"invalid byte-buffer offset negative value -1",
		},
		{
			"tensors data-offsets do not begin at 0",
			Header{Tensors: TensorMap{
				"a": Tensor{
					Name:        "a",
					DType:       dtype.U8,
					Shape:       Shape{2, 2},
					DataOffsets: DataOffsets{1, 5},
				},
			}},
			`invalid tensor "a": expected data-offsets begin 0, actual 1`,
		},
		{
			"tensor names mismatch",
			Header{Tensors: TensorMap{
				"a": Tensor{
					Name:        "b",
					DType:       dtype.U8,
					Shape:       Shape{2, 2},
					DataOffsets: DataOffsets{0, 4},
				},
			}},
			`tensor names mismatch: TensorMap key "a", Tensor.Name "b"`,
		},
		{
			"hole between tensor data-offsets",
			Header{Tensors: TensorMap{
				"a": Tensor{
					Name:        "a",
					DType:       dtype.U8,
					Shape:       Shape{2, 2},
					DataOffsets: DataOffsets{0, 4},
				},
				"b": Tensor{
					Name:        "b",
					DType:       dtype.U8,
					Shape:       Shape{2, 2},
					DataOffsets: DataOffsets{5, 9},
				},
			}},
			`invalid tensor "b": expected data-offsets begin 4, actual 5`,
		},
		{
			"data-offsets end < begin",
			Header{Tensors: TensorMap{
				"a": Tensor{
					Name:        "a",
					DType:       dtype.U8,
					Shape:       Shape{2, 2},
					DataOffsets: DataOffsets{0, 4},
				},
				"b": Tensor{
					Name:        "b",
					DType:       dtype.U8,
					Shape:       Shape{2, 2},
					DataOffsets: DataOffsets{4, 3},
				},
			}},
			`invalid tensor "b": expected data-offsets end >= 4 (begin), actual 3`,
		},
		{
			"data-offsets byte size > shape*dtype byte size",
			Header{Tensors: TensorMap{
				"a": Tensor{
					Name:        "a",
					DType:       dtype.U8,
					Shape:       Shape{2, 2},
					DataOffsets: DataOffsets{0, 5},
				},
			}},
			`invalid tensor "a": byte size computed from shape (4) differs from data-offsets size (5)`,
		},
		{
			"data-offsets byte size < shape*dtype byte size",
			Header{Tensors: TensorMap{
				"a": Tensor{
					Name:        "a",
					DType:       dtype.U8,
					Shape:       Shape{2, 2},
					DataOffsets: DataOffsets{0, 3},
				},
			}},
			`invalid tensor "a": byte size computed from shape (4) differs from data-offsets size (3)`,
		},
		{
			"invalid DType",
			Header{Tensors: TensorMap{
				"a": Tensor{
					Name:        "a",
					DType:       dtype.DType(0),
					Shape:       Shape{2, 2},
					DataOffsets: DataOffsets{0, 4},
				},
			}},
			`invalid tensor "a": invalid DType(0)`,
		},
		{
			"shape contains a negative value",
			Header{Tensors: TensorMap{
				"a": Tensor{
					Name:        "a",
					DType:       dtype.U8,
					Shape:       Shape{2, -1},
					DataOffsets: DataOffsets{0, 2},
				},
			}},
			`invalid tensor "a": shape contains negative value -1`,
		},
		{
			"shape product overflow",
			Header{Tensors: TensorMap{
				"a": Tensor{
					Name:        "a",
					DType:       dtype.U8,
					Shape:       Shape{math.MaxInt, math.MaxInt},
					DataOffsets: DataOffsets{0, 0},
				},
			}},
			`invalid tensor "a": int overflow computing tensor elements size from shape`,
		},
		{
			"shape product * dtype overflow",
			Header{Tensors: TensorMap{
				"a": Tensor{
					Name:        "a",
					DType:       dtype.U16,
					Shape:       Shape{math.MaxInt, 2},
					DataOffsets: DataOffsets{0, 0},
				},
			}},
			`invalid tensor "a": int overflow computing tensor byte size from shape`,
		},
		{
			"shape product * dtype > max int",
			Header{Tensors: TensorMap{
				"a": Tensor{
					Name:        "a",
					DType:       dtype.U8,
					Shape:       Shape{math.MaxInt, 2},
					DataOffsets: DataOffsets{0, 0},
				},
			}},
			`invalid tensor "a": tensor byte size computed from shape is too large for int type: 18446744073709551614`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.h.Validate()
			assert.EqualError(t, err, tc.errMsg)
		})
	}
}
