// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package header

import (
	"encoding/json"
	"testing"

	"github.com/nlpodyssey/safetensors/dtype"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHeader_UnmarshalJSON(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		data := []byte(`{"foo": {"dtype": "U8", "shape": [2, 3], "data_offsets": [0, 6]},` +
			`"bar": {"dtype": "I8", "shape": [4, 5], "data_offsets": [6, 26]},` +
			`"__metadata__": {"foo": "bar", "baz": "qux"}}`)

		var h Header
		err := h.UnmarshalJSON(data)
		require.NoError(t, err)

		expected := Header{
			Metadata: Metadata{"foo": "bar", "baz": "qux"},
			Tensors: TensorMap{
				"foo": Tensor{Name: "foo", DType: dtype.U8, Shape: Shape{2, 3}, DataOffsets: DataOffsets{Begin: 0, End: 6}},
				"bar": Tensor{Name: "bar", DType: dtype.I8, Shape: Shape{4, 5}, DataOffsets: DataOffsets{Begin: 6, End: 26}},
			},
			ByteBufferOffset: 0,
		}
		assert.Equal(t, expected, h)
	})

	t.Run("invalid JSON", func(t *testing.T) {
		var h Header
		err := h.UnmarshalJSON([]byte("{}oh!"))
		require.Error(t, err)
	})

	t.Run("invalid header content", func(t *testing.T) {
		var h Header
		err := h.UnmarshalJSON([]byte(`{"foo": {"bar": "baz"}}`))
		require.Error(t, err)
	})
}

func TestHeader_MarshalJSON(t *testing.T) {
	type m = map[string]any
	type mm = map[string]m

	testCases := []struct {
		name     string
		header   Header
		expected mm
	}{
		{"empty header", Header{}, mm{}},
		{"empty metadata", Header{Metadata: Metadata{}}, mm{}},
		{"empty tensors", Header{Tensors: TensorMap{}}, mm{}},
		{
			"metadata",
			Header{Metadata: Metadata{"foo": "bar"}},
			mm{"__metadata__": m{"foo": "bar"}},
		},
		{
			"tensors",
			Header{Tensors: TensorMap{
				"foo": Tensor{Name: "foo", DType: dtype.U8, Shape: Shape{2, 3}, DataOffsets: DataOffsets{Begin: 0, End: 6}},
				"bar": Tensor{Name: "bar", DType: dtype.I8, Shape: Shape{4, 5}, DataOffsets: DataOffsets{Begin: 6, End: 26}},
			}},
			mm{
				"foo": m{"dtype": "U8", "shape": []any{2.0, 3.0}, "data_offsets": []any{0.0, 6.0}},
				"bar": m{"dtype": "I8", "shape": []any{4.0, 5.0}, "data_offsets": []any{6.0, 26.0}},
			},
		},
		{
			"tensors and metadata",
			Header{
				Metadata: Metadata{"foo": "bar"},
				Tensors: TensorMap{
					"foo": Tensor{Name: "foo", DType: dtype.U8, Shape: Shape{2, 3}, DataOffsets: DataOffsets{Begin: 0, End: 6}},
					"bar": Tensor{Name: "bar", DType: dtype.I8, Shape: Shape{4, 5}, DataOffsets: DataOffsets{Begin: 6, End: 26}},
				},
			},
			mm{
				"__metadata__": m{"foo": "bar"},
				"foo":          m{"dtype": "U8", "shape": []any{2.0, 3.0}, "data_offsets": []any{0.0, 6.0}},
				"bar":          m{"dtype": "I8", "shape": []any{4.0, 5.0}, "data_offsets": []any{6.0, 26.0}},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := tc.header.MarshalJSON()
			require.NoError(t, err)

			var actual mm
			require.NoError(t, json.Unmarshal(b, &actual))
			assert.Equal(t, tc.expected, actual)
		})
	}
}