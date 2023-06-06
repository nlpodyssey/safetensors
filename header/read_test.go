// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package header

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"
	"testing/iotest"

	"github.com/nlpodyssey/safetensors/dtype"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRead_Success(t *testing.T) {
	testCases := []struct {
		name string
		json string
		want Header
	}{
		{
			"empty object",
			`{}`,
			Header{},
		},
		{
			"empty metadata",
			`{"__metadata__": {}}`,
			Header{},
		},
		{
			"metadata",
			`{"__metadata__": {"foo": "bar", "baz": "qux"}}`,
			Header{Metadata: Metadata{"foo": "bar", "baz": "qux"}},
		},
		{
			"tensors",
			`{"foo": {"dtype": "U8", "shape": [2, 3], "data_offsets": [0, 6]},` +
				`"bar": {"dtype": "I8", "shape": [4, 5], "data_offsets": [6, 26]}}`,
			Header{Tensors: TensorMap{
				"foo": Tensor{Name: "foo", DType: dtype.U8, Shape: Shape{2, 3}, DataOffsets: DataOffsets{Begin: 0, End: 6}},
				"bar": Tensor{Name: "bar", DType: dtype.I8, Shape: Shape{4, 5}, DataOffsets: DataOffsets{Begin: 6, End: 26}},
			}},
		},
		{
			"tensors and metadata",
			`{"foo": {"dtype": "U8", "shape": [2, 3], "data_offsets": [0, 6]},` +
				`"bar": {"dtype": "I8", "shape": [4, 5], "data_offsets": [6, 26]},` +
				`"__metadata__": {"foo": "bar", "baz": "qux"}}`,
			Header{
				Metadata: Metadata{"foo": "bar", "baz": "qux"},
				Tensors: TensorMap{
					"foo": Tensor{Name: "foo", DType: dtype.U8, Shape: Shape{2, 3}, DataOffsets: DataOffsets{Begin: 0, End: 6}},
					"bar": Tensor{Name: "bar", DType: dtype.I8, Shape: Shape{4, 5}, DataOffsets: DataOffsets{Begin: 6, End: 26}},
				},
			},
		},
		{
			"padding before and after",
			" \n\r\t" + `{"foo": {"dtype": "U8", "shape": [2, 3], "data_offsets": [0, 6]},` +
				`"__metadata__": {"foo": "bar"}}` + " \n\r\t",
			Header{
				Metadata: Metadata{"foo": "bar"},
				Tensors: TensorMap{
					"foo": Tensor{Name: "foo", DType: dtype.U8, Shape: Shape{2, 3}, DataOffsets: DataOffsets{Begin: 0, End: 6}},
				},
			},
		},
	}

	for _, tc := range testCases {
		want := tc.want
		want.ByteBufferOffset = 8 + len(tc.json)

		for _, byteBufferSize := range []int{0, 100} {
			t.Run(fmt.Sprintf("%s plus %d bytes", tc.name, byteBufferSize), func(t *testing.T) {
				data := makeData(tc.json, byteBufferSize)
				h, err := Read(bytes.NewReader(data))
				require.NoError(t, err)
				assert.Equal(t, want, h)
			})
		}
	}
}

func TestRead_Failure(t *testing.T) {
	testCases := []struct {
		name   string
		json   string
		errMsg string
	}{
		{"size 0", "", "header size too small: 0"},
		{"size 1", " ", "header size too small: 1"},
		{
			"bad trailing data, valid JSON token", "{}9",
			"failed to JSON-decode header: unexpected data at byte offset 2",
		},
		{
			"bad trailing data, invalid JSON token", "{}~",
			"failed to JSON-decode header: invalid character '~' looking for beginning of value",
		},
		{
			"incomplete JSON", `{"foo`,
			"failed to JSON-decode header: unexpected EOF",
		},
		{
			"bad JSON", `{1: 2}`,
			"failed to JSON-decode header: invalid character '1' looking for beginning of object key string",
		},
		{
			"bad metadata", `{"__metadata__": {"foo": 1}}`,
			`failed to interpret header metadata: found non-string value for key "foo"`,
		},
		{
			"dtype missing",
			`{"foo": {"shape": [2, 3], "data_offsets": [0, 6]}}`,
			`failed to interpret header tensor "foo": "dtype" is missing`,
		},
		{
			"shape missing",
			`{"foo": {"dtype": "U8", "data_offsets": [0, 6]}}`,
			`failed to interpret header tensor "foo": "shape" is missing`,
		},
		{
			"data_offsets missing",
			`{"foo": {"dtype": "U8", "shape": [2, 3]}}`,
			`failed to interpret header tensor "foo": "data_offsets" is missing`,
		},
		{
			"unknown tensor key",
			`{"foo": {"dtype": "U8", "shape": [2, 3], "data_offsets": [0, 6], "bar": "baz"}}`,
			`failed to interpret header tensor "foo": JSON object contains unknown keys`,
		},
		{
			"dtype is not string",
			`{"foo": {"dtype": 123, "shape": [2, 3], "data_offsets": [0, 6]}}`,
			`failed to interpret header tensor "foo": found non-string "dtype" value`,
		},
		{
			"invalid dtype",
			`{"foo": {"dtype": "X9", "shape": [2, 3], "data_offsets": [0, 6]}}`,
			`failed to interpret header tensor "foo": invalid "dtype" value: "X9"`,
		},
		{
			"shape is not array",
			`{"foo": {"dtype": "U8", "shape": 123, "data_offsets": [0, 6]}}`,
			`failed to interpret header tensor "foo": found non-array "shape" value`,
		},
		{
			"shape item is not number",
			`{"foo": {"dtype": "U8", "shape": [2, "3"], "data_offsets": [0, 6]}}`,
			`failed to interpret header tensor "foo": ` +
				`failed to interpret "shape" value at index 1: ` +
				`value is not a number`,
		},
		{
			"shape item is float with fraction",
			`{"foo": {"dtype": "U8", "shape": [2, 3.0], "data_offsets": [0, 6]}}`,
			`failed to interpret header tensor "foo": ` +
				`failed to interpret "shape" value at index 1: ` +
				`failed to convert value "3.0" to int: ` +
				`strconv.ParseInt: parsing "3.0": invalid syntax`,
		},
		{
			"shape item is float with exponent",
			`{"foo": {"dtype": "U8", "shape": [2, 3e1], "data_offsets": [0, 6]}}`,
			`failed to interpret header tensor "foo": ` +
				`failed to interpret "shape" value at index 1: ` +
				`failed to convert value "3e1" to int: ` +
				`strconv.ParseInt: parsing "3e1": invalid syntax`,
		},
		{
			"shape item int too big",
			`{"foo": {"dtype": "U8", "shape": [2, 18446744073709551615], "data_offsets": [0, 6]}}`,
			`failed to interpret header tensor "foo": ` +
				`failed to interpret "shape" value at index 1: ` +
				`failed to convert value "18446744073709551615" to int: ` +
				`strconv.ParseInt: parsing "18446744073709551615": value out of range`,
		},
		{
			"shape item is negative",
			`{"foo": {"dtype": "U8", "shape": [2, -1], "data_offsets": [0, 6]}}`,
			`failed to interpret header tensor "foo": ` +
				`failed to interpret "shape" value at index 1: ` +
				`value is negative: -1`,
		},
		{
			"data_offsets is not array",
			`{"foo": {"dtype": "U8", "shape": [2, 3], "data_offsets": 123}}`,
			`failed to interpret header tensor "foo": found non-array "data_offsets" value`,
		},
		{
			"data_offsets len is not 2",
			`{"foo": {"dtype": "U8", "shape": [2, 3], "data_offsets": [1, 2, 3]}}`,
			`failed to interpret header tensor "foo": bad "data_offsets" length: expected 2, actual 3`,
		},
		{
			"data_offsets item is not number",
			`{"foo": {"dtype": "U8", "shape": [2, 3], "data_offsets": [0, "6"]}}`,
			`failed to interpret header tensor "foo": ` +
				`failed to interpret "data_offsets" value at index 1: ` +
				`value is not a number`,
		},
		{
			"data_offsets item is float with fraction",
			`{"foo": {"dtype": "U8", "shape": [2, 3], "data_offsets": [0, 6.0]}}`,
			`failed to interpret header tensor "foo": ` +
				`failed to interpret "data_offsets" value at index 1: ` +
				`failed to convert value "6.0" to int: ` +
				`strconv.ParseInt: parsing "6.0": invalid syntax`,
		},
		{
			"data_offsets item is float with exponent",
			`{"foo": {"dtype": "U8", "shape": [2, 3], "data_offsets": [0, 6e1]}}`,
			`failed to interpret header tensor "foo": ` +
				`failed to interpret "data_offsets" value at index 1: ` +
				`failed to convert value "6e1" to int: ` +
				`strconv.ParseInt: parsing "6e1": invalid syntax`,
		},
		{
			"data_offsets item int too big",
			`{"foo": {"dtype": "U8", "shape": [2, 3], "data_offsets": [0, 18446744073709551615]}}`,
			`failed to interpret header tensor "foo": ` +
				`failed to interpret "data_offsets" value at index 1: ` +
				`failed to convert value "18446744073709551615" to int: ` +
				`strconv.ParseInt: parsing "18446744073709551615": value out of range`,
		},
		{
			"data_offsets item is negative",
			`{"foo": {"dtype": "U8", "shape": [2, 3], "data_offsets": [0, -1]}}`,
			`failed to interpret header tensor "foo": ` +
				`failed to interpret "data_offsets" value at index 1: ` +
				`value is negative: -1`,
		},
	}

	for _, tc := range testCases {
		for _, byteBufferSize := range []int{0, 100} {
			t.Run(fmt.Sprintf("%s plus %d bytes", tc.name, byteBufferSize), func(t *testing.T) {
				data := makeData(tc.json, 0)
				h, err := Read(bytes.NewReader(data))
				require.EqualError(t, err, tc.errMsg)
				assert.Equal(t, Header{}, h)
			})
		}
	}

	t.Run("size too large", func(t *testing.T) {
		data := []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff} // max uint64
		h, err := Read(bytes.NewReader(data))
		require.EqualError(t, err, "header size too large: 18446744073709551615")
		assert.Equal(t, Header{}, h)
	})

	t.Run("reader error reading size", func(t *testing.T) {
		data := []byte{2, 0, 0, 0, 0, 0, 0} // one byte is missing
		h, err := Read(iotest.DataErrReader(bytes.NewReader(data)))
		require.EqualError(t, err, "failed to read header size: unexpected EOF")
		assert.Equal(t, Header{}, h)
	})

	t.Run("reader error reading JSON", func(t *testing.T) {
		data := makeData(`{"foo`, 0)
		h, err := Read(iotest.DataErrReader(bytes.NewReader(data)))
		require.EqualError(t, err, "failed to JSON-decode header: unexpected EOF")
		assert.Equal(t, Header{}, h)
	})
}

func makeData(json string, byteBufferSize int) []byte {
	data := make([]byte, 8+len(json)+byteBufferSize)
	binary.LittleEndian.PutUint64(data, uint64(len(json)))
	copy(data[8:len(json)+8], json)
	for i := len(json) + 8; i < len(data); i++ {
		data[i] = 0xff
	}
	return data
}
