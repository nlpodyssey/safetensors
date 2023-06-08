// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/nlpodyssey/safetensors/dtype"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var _ SerializableTensor = Tensor{}

func TestTensor_WriteTo(t *testing.T) {
	for name, def := range commonDefinitions {
		t.Run(name, func(t *testing.T) {
			tensor, err := NewTensor(name, def.dType, def.shape, def.typedValue)
			require.NoError(t, err)

			var buf bytes.Buffer
			n, err := tensor.WriteTo(&buf)
			require.NoError(t, err)
			assert.Equal(t, int64(len(def.bytes)), n)
			assert.Equal(t, def.bytes, buf.Bytes())
		})
	}
}

func TestNewTensor(t *testing.T) {
	for name, def := range commonDefinitions {
		t.Run(name, func(t *testing.T) {
			tensor, err := NewTensor(name, def.dType, def.shape, def.typedValue)
			require.NoError(t, err)
			assert.Equal(t, name, tensor.Name())
			assert.Equal(t, def.dType, tensor.DType())
			if len(def.shape) == 0 {
				assert.Nil(t, tensor.Shape())
			} else {
				assert.Equal(t, def.shape, tensor.Shape())
			}
			assert.Equal(t, def.typedValue, tensor.Data())
		})
	}

	allDTypes := []dtype.DType{
		dtype.Bool, dtype.U8, dtype.I8,
		dtype.U16, dtype.I16, dtype.F16, dtype.BF16,
		dtype.U32, dtype.I32, dtype.F32,
		dtype.U64, dtype.I64, dtype.F64,
	}
	for _, dt := range allDTypes {
		t.Run(fmt.Sprintf("dtype %s allows nil data", dt), func(t *testing.T) {
			tensor, err := NewTensor("", dt, []int{0}, nil)
			require.NoError(t, err)
			assert.Nil(t, tensor.Data())
		})
	}

	errorTestCases := []struct {
		name   string
		dType  dtype.DType
		shape  []int
		data   any
		errMsg string
	}{
		{
			"invalid dType", dtype.DType(42), []int{0}, nil,
			"invalid or unsupported DType: invalid DType(42)",
		},
		{
			"type mismatch", dtype.U16, []int{1}, []uint8{42},
			"expected DType U16 to match data type []uint16, actual data type []uint8",
		},
		{
			"negative value in size", dtype.U8, []int{1, -1}, []uint8{1},
			"shape contains a negative value",
		},
		{
			"shape and len mismatch", dtype.U8, []int{2, 3}, []uint8{1, 2, 3, 4},
			"the size computed from shape (6) does not match data length (4)",
		},
	}

	for _, tc := range errorTestCases {
		t.Run(tc.name, func(t *testing.T) {
			tensor, err := NewTensor(tc.name, tc.dType, tc.shape, tc.data)
			require.EqualError(t, err, tc.errMsg)
			assert.Equal(t, Tensor{}, tensor)
		})
	}
}
