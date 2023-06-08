// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package header

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDataOffsets_UnmarshalJSON(t *testing.T) {
	t.Run("valid value", func(t *testing.T) {
		var d DataOffsets
		assert.NoError(t, d.UnmarshalJSON([]byte("[1, 2]")))
		assert.Equal(t, DataOffsets{Begin: 1, End: 2}, d)
	})

	t.Run("invalid values", func(t *testing.T) {
		values := []string{"null", "[]", "[1]", "[1,2,3]", "["}
		for _, v := range values {
			var d DataOffsets
			assert.Error(t, d.UnmarshalJSON([]byte(v)), v)
		}
	})
}

func TestDataOffsets_MarshalJSON(t *testing.T) {
	d := DataOffsets{Begin: 1, End: 2}
	b, err := d.MarshalJSON()
	assert.NoError(t, err)
	assert.Equal(t, "[1,2]", string(b))
}
