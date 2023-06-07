// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"testing"

	"github.com/nlpodyssey/safetensors/dtype"
	"github.com/nlpodyssey/safetensors/float16"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewLazy(t *testing.T) {
	t.Run("header size limit", func(t *testing.T) {
		data := makeData(`{"__metadata__":{}}`, nil)
		st, err := NewLazy(bytes.NewReader(data), 10)
		require.EqualError(t, err, "failed to read safetensors header: failed to JSON-decode header: unexpected EOF")
		require.ErrorIs(t, err, io.ErrUnexpectedEOF)
		require.Nil(t, st)
	})

	t.Run("error reading header size", func(t *testing.T) {
		data := []byte{0}
		st, err := NewLazy(bytes.NewReader(data), 10)
		require.EqualError(t, err, "failed to read safetensors header: failed to read header size: unexpected EOF")
		require.ErrorIs(t, err, io.ErrUnexpectedEOF)
		require.Nil(t, st)
	})
}

func TestLazyST_Metadata(t *testing.T) {
	testCases := []struct {
		jsonHeader string
		want       map[string]string
	}{
		{`{}`, nil},
		{`{"__metadata__": {}}`, nil},
		{`{"__metadata__": {"foo": "bar"}}`, map[string]string{"foo": "bar"}},
	}
	for _, tc := range testCases {
		t.Run(tc.jsonHeader, func(t *testing.T) {
			data := makeData(tc.jsonHeader, nil)
			st, err := NewLazy(bytes.NewReader(data), 0)
			require.NoError(t, err)
			assert.Equal(t, tc.want, st.Metadata())
		})
	}
}

func TestLazyST_TensorNames(t *testing.T) {
	testCases := []struct {
		jsonHeader string
		want       []string
	}{
		{`{}`, nil},
		{`{"a":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}`, []string{"a"}},
		{
			`{"a":{"dtype":"U8","shape":[0],"data_offsets":[0,0]},` +
				`"b":{"dtype":"I8","shape":[0],"data_offsets":[0,0]}}`,
			[]string{"a", "b"},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.jsonHeader, func(t *testing.T) {
			data := makeData(tc.jsonHeader, nil)
			st, err := NewLazy(bytes.NewReader(data), 0)
			require.NoError(t, err)
			names := st.TensorNames()
			require.Len(t, names, len(tc.want))
			for _, n := range tc.want {
				assert.Contains(t, names, n)
			}
		})
	}
}

func TestLazyST_LazyTensor(t *testing.T) {
	definitions := map[string]struct {
		dType      dtype.DType
		shape      []int
		typedValue any
		bytes      []byte
	}{
		"bool": {
			dtype.Bool, []int{2, 2},
			[]bool{false, true, true, true},
			[]byte{0x00, 0x01, 0xfe, 0xff},
		},
		"u8": {
			dtype.U8, []int{2, 2},
			[]uint8{0, 1, 254, 255},
			[]byte{0x00, 0x01, 0xfe, 0xff},
		},
		"i8": {
			dtype.I8, []int{2, 2},
			[]int8{0, 1, -2, -1},
			[]byte{0x00, 0x01, 0xfe, 0xff},
		},
		"u16": {
			dtype.U16, []int{2, 2},
			[]uint16{0, 1, 65534, 65535},
			[]byte{
				0x00, 0x00 /**/, 0x01, 0x00,
				0xfe, 0xff /**/, 0xff, 0xff,
			},
		},
		"i16": {
			dtype.I16, []int{2, 2},
			[]int16{0, 1, -2, -1},
			[]byte{
				0x00, 0x00 /**/, 0x01, 0x00,
				0xfe, 0xff /**/, 0xff, 0xff,
			},
		},
		"f16": {
			dtype.F16, []int{2, 2},
			[]float16.F16{0x0001, 0x0203, 0x0405, 0x0607},
			[]byte{
				0x01, 0x00 /**/, 0x03, 0x02,
				0x05, 0x04 /**/, 0x07, 0x06,
			},
		},
		"bf16": {
			dtype.BF16, []int{2, 2},
			[]float16.BF16{0x0001, 0x0203, 0x0405, 0x0607},
			[]byte{
				0x01, 0x00 /**/, 0x03, 0x02,
				0x05, 0x04 /**/, 0x07, 0x06,
			},
		},
		"u32": {
			dtype.U32, []int{2, 2},
			[]uint32{1, 2, 4294967294, 4294967295},
			[]byte{
				0x01, 0x00, 0x00, 0x00 /**/, 0x02, 0x00, 0x00, 0x00,
				0xfe, 0xff, 0xff, 0xff /**/, 0xff, 0xff, 0xff, 0xff,
			},
		},
		"i32": {
			dtype.I32, []int{2, 2},
			[]int32{1, 2, -2, -1},
			[]byte{
				0x01, 0x00, 0x00, 0x00 /**/, 0x02, 0x00, 0x00, 0x00,
				0xfe, 0xff, 0xff, 0xff /**/, 0xff, 0xff, 0xff, 0xff,
			},
		},
		"f32": {
			dtype.F32, []int{2, 2},
			[]float32{1, 2, -1, -2},
			[]byte{
				0x00, 0x00, 0x80, 0x3f /**/, 0x00, 0x00, 0x00, 0x40,
				0x00, 0x00, 0x80, 0xbf /**/, 0x00, 0x00, 0x00, 0xc0,
			},
		},
		"u64": {
			dtype.U64, []int{2, 1},
			[]uint64{1, 18446744073709551615},
			[]byte{
				0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
			},
		},
		"i64": {
			dtype.I64, []int{1, 2},
			[]int64{1, -1},
			[]byte{
				0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
			},
		},
		"f64": {
			dtype.F64, []int{2},
			[]float64{1, -1},
			[]byte{
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0xbf,
			},
		},
		"zero data": {
			dtype.U8, []int{0},
			[]uint8{},
			nil,
		},
		"no shape scalar": {
			dtype.U8, nil,
			[]uint8{42},
			[]byte{42},
		},
	}
	header := make(map[string]map[string]any, len(definitions))
	byteBuffer := bytes.NewBuffer(nil)

	for name, def := range definitions {
		shape := def.shape
		if shape == nil {
			shape = []int{}
		}
		header[name] = map[string]any{
			"dtype":        def.dType.String(),
			"shape":        shape,
			"data_offsets": [2]int{byteBuffer.Len(), byteBuffer.Len() + len(def.bytes)},
		}
		_, err := byteBuffer.Write(def.bytes)
		require.NoError(t, err)
	}

	jsonHeader, err := json.Marshal(header)
	require.NoError(t, err)

	sfData := makeData(string(jsonHeader), byteBuffer.Bytes())

	st, err := NewLazy(bytes.NewReader(sfData), 0)
	require.NoError(t, err)

	t.Run("tensor not found", func(t *testing.T) {
		lt, ok := st.LazyTensor("foo")
		assert.False(t, ok)
		assert.Equal(t, LazyTensor{}, lt)
	})

	allTensors, err := st.AllTensors()
	require.NoError(t, err)
	assert.Len(t, allTensors, len(definitions))

	for name, def := range definitions {
		t.Run(fmt.Sprintf("tensor %q", name), func(t *testing.T) {
			lt, ok := st.LazyTensor(name)
			assert.True(t, ok)

			assert.Equal(t, name, lt.Name())
			assert.Equal(t, def.dType, lt.DType())
			assert.Equal(t, def.shape, lt.Shape())

			tensor, err := lt.Tensor()
			require.NoError(t, err)
			assert.Equal(t, name, tensor.Name())
			assert.Equal(t, def.dType, tensor.DType())
			assert.Equal(t, def.shape, tensor.Shape())
			assert.Equal(t, def.typedValue, tensor.Data())

			assert.Contains(t, allTensors, tensor)

			rawTensor, err := lt.RawTensor()
			require.NoError(t, err)
			assert.Equal(t, name, rawTensor.Name())
			assert.Equal(t, def.dType, rawTensor.DType())
			assert.Equal(t, def.shape, rawTensor.Shape())
			assert.Equal(t, def.bytes, rawTensor.Data())

			data, err := lt.ReadData()
			require.NoError(t, err)
			assert.Equal(t, def.bytes, data)

			buf := bytes.Buffer{}
			n, err := lt.ReadAndCopyData(&buf)
			require.NoError(t, err)
			assert.Equal(t, int64(len(def.bytes)), n)
			assert.Equal(t, def.bytes, buf.Bytes())
		})
	}
}

func makeData(jsonHeader string, byteBuffer []byte) []byte {
	data := make([]byte, 8+len(jsonHeader)+len(byteBuffer))
	binary.LittleEndian.PutUint64(data, uint64(len(jsonHeader)))
	copy(data[8:8+len(jsonHeader)], jsonHeader)
	copy(data[8+len(jsonHeader):], byteBuffer)
	return data
}
