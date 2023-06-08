// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var _ SerializableTensor = RawTensor{}

func TestRawTensor_WriteTo(t *testing.T) {
	for name, def := range commonDefinitions {
		t.Run(name, func(t *testing.T) {
			rt := RawTensor{
				name:  name,
				dType: def.dType,
				shape: def.shape,
				data:  def.bytes,
			}

			var buf bytes.Buffer
			n, err := rt.WriteTo(&buf)
			require.NoError(t, err)
			assert.Equal(t, int64(len(def.bytes)), n)
			assert.Equal(t, def.bytes, buf.Bytes())
		})
	}
}
