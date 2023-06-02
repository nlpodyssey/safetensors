// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import "fmt"

// checkedMul multiplies a and b and checks for overflow.
func checkedMul(a, b uint64) (uint64, error) {
	c := a * b
	if a > 1 && b > 1 && c/a != b {
		return c, fmt.Errorf("multiplication overflow: %d * %d", a, b)
	}
	return c, nil
}
