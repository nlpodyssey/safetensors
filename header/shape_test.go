// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package header

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

var _ json.Marshaler = Shape{}

func TestShape_MarshalJSON(t *testing.T) {
	testCases := []struct {
		shape Shape
		json  string
	}{
		{nil, "[]"},
		{Shape{}, "[]"},
		{Shape{42}, "[42]"},
	}
	for _, tc := range testCases {
		b, err := tc.shape.MarshalJSON()
		assert.NoError(t, err, tc)
		assert.Equal(t, tc.json, string(b), tc)
	}
}
