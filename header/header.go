// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package header

import (
	"bytes"
	"encoding/json"
)

// Header provides tensors information and metadata, as defined by
// the safetensors format.
type Header struct {
	Tensors  TensorMap
	Metadata Metadata
	// ByteBufferOffset indicates the byte index position where the byte-buffer
	// is expected to start, relative to the beginning of the whole
	// safetensors data stream (or file).
	ByteBufferOffset int
}

// Metadata is a set of free-form key/value string pairs.
type Metadata map[string]string

// UnmarshalJSON decodes a safetensors JSON header.
// The resulting Header is not validated.
// Header.ByteBufferOffset is always zero.
func (h *Header) UnmarshalJSON(b []byte) error {
	r := bytes.NewReader(b)
	raw, err := readAndDecodeJSON(r, int64(len(b)))
	if err != nil {
		return err
	}
	if *h, err = convertRawHeader(raw); err != nil {
		return err
	}
	return nil
}

// MarshalJSON encodes the header to safetensors format.
// The value of Header.ByteBufferOffset is ignored.
func (h Header) MarshalJSON() ([]byte, error) {
	m := make(map[string]any, len(h.Tensors)+1)

	if len(h.Metadata) > 0 {
		m["__metadata__"] = h.Metadata
	}

	for name, t := range h.Tensors {
		m[name] = t
	}

	b, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}
	return b, nil
}
