// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package header

import "encoding/json"

// The Shape of a tensor.
type Shape []int

// MarshalJSON prevents a nil Shape to be serialized as "null",
// preferring an empty array "[]" instead. This allows the JSON
// value to be compliant with safetensors format.
func (s Shape) MarshalJSON() ([]byte, error) {
	if s == nil {
		return []byte("[]"), nil
	}
	return json.Marshal([]int(s))
}
