// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package header

import (
	"encoding/json"
	"fmt"
)

// DataOffsets describes "[Begin, End)" byte range of the tensor's data
// within the safetensors byte-buffer.
//
// Tensor data starts at Begin byte index (inclusive) and ends at End byte
// index (exclusive). Both positions are relative to the beginning of the
// byte-buffer.
type DataOffsets struct {
	// Begin is the lower bound byte index (included).
	Begin int
	// End is the upper bound byte index (excluded).
	End int
}

// UnmarshalJSON deserializes a DataOffsets object from the JSON
// value expected from safetensors format (that is, an array of two numbers).
func (a *DataOffsets) UnmarshalJSON(b []byte) error {
	var decoded []int
	if err := json.Unmarshal(b, &decoded); err != nil {
		return err
	}
	if len(decoded) != 2 {
		return fmt.Errorf("invalid data-offsets value: %q", string(b))
	}
	*a = DataOffsets{
		Begin: decoded[0],
		End:   decoded[1],
	}
	return nil
}

// MarshalJSON serializes a DataOffsets object to a value appropriate for
// safetensors format (that is, an array of two numbers).
func (a DataOffsets) MarshalJSON() ([]byte, error) {
	return json.Marshal([2]int{a.Begin, a.End})
}
