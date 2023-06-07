// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"bytes"
	"fmt"
	"io"
	"testing"

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
	sfData := makeCommonData(t)

	st, err := NewLazy(bytes.NewReader(sfData), 0)
	require.NoError(t, err)

	t.Run("tensor not found", func(t *testing.T) {
		lt, ok := st.LazyTensor("foo")
		assert.False(t, ok)
		assert.Equal(t, LazyTensor{}, lt)
	})

	allTensors, err := st.AllTensors()
	require.NoError(t, err)
	assert.Len(t, allTensors, len(commonDefinitions))

	allRawTensors, err := st.AllRawTensors()
	require.NoError(t, err)
	assert.Len(t, allRawTensors, len(commonDefinitions))

	for name, def := range commonDefinitions {
		t.Run(fmt.Sprintf("tensor %q", name), func(t *testing.T) {
			lt, ok := st.LazyTensor(name)
			assert.True(t, ok)

			assert.Equal(t, name, lt.Name())
			assert.Equal(t, def.dType, lt.DType())
			if len(def.shape) == 0 {
				assert.Nil(t, lt.Shape())
			} else {
				assert.Equal(t, def.shape, lt.Shape())
			}

			tensor, err := lt.Tensor()
			require.NoError(t, err)
			assert.Equal(t, name, tensor.Name())
			assert.Equal(t, def.dType, tensor.DType())
			if len(def.shape) == 0 {
				assert.Nil(t, tensor.Shape())
			} else {
				assert.Equal(t, def.shape, tensor.Shape())
			}
			assert.Equal(t, def.typedValue, tensor.Data())

			assert.Contains(t, allTensors, tensor)

			rawTensor, err := lt.RawTensor()
			require.NoError(t, err)
			assert.Equal(t, name, rawTensor.Name())
			assert.Equal(t, def.dType, rawTensor.DType())
			if len(def.shape) == 0 {
				assert.Nil(t, rawTensor.Shape())
			} else {
				assert.Equal(t, def.shape, rawTensor.Shape())
			}
			assert.Equal(t, def.bytes, rawTensor.Data())

			assert.Contains(t, allRawTensors, rawTensor)

			data, err := lt.ReadData()
			require.NoError(t, err)
			assert.Equal(t, def.bytes, data)

			buf := bytes.Buffer{}
			n, err := lt.WriteTo(&buf)
			require.NoError(t, err)
			assert.Equal(t, int64(len(def.bytes)), n)
			assert.Equal(t, def.bytes, buf.Bytes())
		})
	}
}
