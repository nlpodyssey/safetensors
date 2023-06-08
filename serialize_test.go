// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSerialize(t *testing.T) {
	var buf bytes.Buffer
	{
		tensors := make([]Tensor, 0, len(commonDefinitions))
		for name, def := range commonDefinitions {
			tensors = append(tensors, Tensor{
				name:  name,
				dType: def.dType,
				shape: def.shape,
				data:  def.typedValue,
			})
		}
		metadata := map[string]string{"meta...": "data!"}

		err := Serialize(&buf, tensors, metadata)
		require.NoError(t, err)
	}

	st, err := ReadAll(&buf, 0)
	require.NoError(t, err)

	assert.Equal(t, map[string]string{"meta...": "data!"}, st.Metadata)
	assert.Len(t, st.Tensors, len(commonDefinitions))

	for name, def := range commonDefinitions {
		t.Run(fmt.Sprintf("tensor %q", name), func(t *testing.T) {
			var tensor *Tensor
			for i := range st.Tensors {
				if st.Tensors[i].Name() == name {
					tensor = &st.Tensors[i]
				}
			}
			if tensor == nil {
				t.Fatal("tensor not found")
			}
			assert.Equal(t, def.dType, tensor.DType())
			assert.Equal(t, def.shape, tensor.Shape())
			assert.Equal(t, def.typedValue, tensor.Data())
		})
	}
}
