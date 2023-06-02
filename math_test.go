// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"math"
	"testing"
)

func Test_CheckedMul(t *testing.T) {
	const max = math.MaxUint64

	t.Run("no overflow", func(t *testing.T) {
		testCases := [][2]uint64{
			{0, 0},
			{0, 1},
			{0, 2},
			{1, 1},
			{1, 2},
			{max, 0},
			{max, 1},
			{max / 2, 2},
		}
		for _, tc := range testCases {
			for _, pair := range [][2]uint64{tc, {tc[1], tc[0]}} {
				want := pair[0] * pair[1]

				c, err := checkedMul(pair[0], pair[1])
				if c != want || err != nil {
					t.Errorf("%d * %d: want (%d, nil), got (%d, %v)", pair[0], pair[1], want, c, err)
				}
			}
		}
	})

	t.Run("overflow", func(t *testing.T) {
		testCases := [][2]uint64{
			{max, 2},
			{max / 2, 3},
			{max, max},
		}
		for _, tc := range testCases {
			for _, pair := range [][2]uint64{tc, {tc[1], tc[0]}} {
				c, err := checkedMul(pair[0], pair[1])
				if err == nil {
					t.Errorf("%d * %d: want error, got (%d, nil)", pair[0], pair[1], c)
				}
			}
		}
	})
}
