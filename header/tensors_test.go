// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package header

import (
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDataOffsets_Less(t *testing.T) {
	dOff := func(b, e int) DataOffsets { return DataOffsets{Begin: b, End: e} }

	testCases := []struct {
		a    DataOffsets
		b    DataOffsets
		want bool
	}{
		{dOff(1, 2), dOff(1, 2), false},
		{dOff(1, 2), dOff(3, 4), true},
		{dOff(3, 4), dOff(1, 2), false},
		{dOff(1, 2), dOff(1, 3), true},
		{dOff(1, 3), dOff(1, 2), false},
	}

	for _, tc := range testCases {
		assert.Equal(t, tc.want, tc.a.Less(tc.b))
	}
}

func TestTensorMap_TensorSlice(t *testing.T) {
	assert.Nil(t, TensorMap(nil).TensorSlice())
	assert.Nil(t, TensorMap{}.TensorSlice())

	tm := TensorMap{
		"foo": Tensor{Name: "foo"},
		"bar": Tensor{Name: "bar"},
		"baz": Tensor{Name: "baz"},
	}
	ts := tm.TensorSlice()
	require.Len(t, ts, 3)
	for _, tensor := range tm {
		assert.Contains(t, ts, tensor)
	}
}

func TestTensorSlice_Len(t *testing.T) {
	assert.Equal(t, 0, TensorSlice(nil).Len())
	assert.Equal(t, 0, TensorSlice{}.Len())
	assert.Equal(t, 1, TensorSlice{Tensor{}}.Len())
	assert.Equal(t, 2, TensorSlice{Tensor{}, Tensor{}}.Len())
}

func TestTensorSlice_Swap(t *testing.T) {
	ts := TensorSlice{Tensor{Name: "a"}, Tensor{Name: "b"}, Tensor{Name: "c"}}

	ts.Swap(1, 2)
	require.Equal(t, TensorSlice{Tensor{Name: "a"}, Tensor{Name: "c"}, Tensor{Name: "b"}}, ts)

	ts.Swap(1, 0)
	require.Equal(t, TensorSlice{Tensor{Name: "c"}, Tensor{Name: "a"}, Tensor{Name: "b"}}, ts)
}

func TestTensorSliceByOffset(t *testing.T) {
	ts := TensorSlice{
		Tensor{Name: "a", DataOffsets: DataOffsets{Begin: 1, End: 2}},
		Tensor{Name: "b", DataOffsets: DataOffsets{Begin: 3, End: 4}},
		Tensor{Name: "c", DataOffsets: DataOffsets{Begin: 2, End: 3}},
		Tensor{Name: "d", DataOffsets: DataOffsets{Begin: 0, End: 1}},
	}
	sort.Sort(TensorSliceByDataOffsets{ts})
	assert.Equal(t, TensorSlice{
		Tensor{Name: "d", DataOffsets: DataOffsets{Begin: 0, End: 1}},
		Tensor{Name: "a", DataOffsets: DataOffsets{Begin: 1, End: 2}},
		Tensor{Name: "c", DataOffsets: DataOffsets{Begin: 2, End: 3}},
		Tensor{Name: "b", DataOffsets: DataOffsets{Begin: 3, End: 4}},
	}, ts)
}
