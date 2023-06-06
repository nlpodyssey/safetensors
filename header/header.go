// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package header

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
