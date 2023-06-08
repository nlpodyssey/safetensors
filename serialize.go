// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/nlpodyssey/safetensors/dtype"
	"github.com/nlpodyssey/safetensors/header"
)

// SerializableTensor is implemented by any tensor object whose data can be
// serialized to safetensors format.
type SerializableTensor interface {
	Name() string
	DType() dtype.DType
	Shape() []int
	io.WriterTo
}

// Serialize the given tensors and additional metadata to safetensors format,
// writing the result to "w".
func Serialize[T SerializableTensor](w io.Writer, tensors []T, metadata map[string]string) error {
	tensorSlice, err := makeHeaderTensorSlice(tensors)
	if err != nil {
		return err
	}
	head, err := makeValidHeader(tensorSlice, metadata)
	if err != nil {
		return err
	}
	if err = writeHeader(w, head); err != nil {
		return err
	}
	return writeTensors(w, tensors, tensorSlice)
}

func makeHeaderTensorSlice[T SerializableTensor](tensors []T) (header.TensorSlice, error) {
	out := make(header.TensorSlice, len(tensors))
	offset := 0
	for i, tensor := range tensors {
		out[i], offset = newHeaderTensor(tensor, offset)
	}
	return out, nil
}

func newHeaderTensor(tensor SerializableTensor, beginOffset int) (_ header.Tensor, endOffset int) {
	dType := tensor.DType()
	shape := tensor.Shape()
	endOffset = beginOffset + shapeToByteSize(shape, dType)
	return header.Tensor{
		Name:  tensor.Name(),
		DType: dType,
		Shape: shape,
		DataOffsets: header.DataOffsets{
			Begin: beginOffset,
			End:   endOffset,
		},
	}, endOffset
}

func shapeToByteSize(shape []int, dType dtype.DType) int {
	size := 1
	for _, v := range shape {
		size *= v
	}
	return size * dType.Size()
}

func makeValidHeader(tensors header.TensorSlice, metadata map[string]string) (header.Header, error) {
	head, err := makeHeader(tensors, metadata)
	if err != nil {
		return header.Header{}, err
	}
	if err = head.Validate(); err != nil {
		return header.Header{}, fmt.Errorf("failed to generate a valid header: %w", err)
	}
	return head, nil
}

func makeHeader(tensors header.TensorSlice, metadata map[string]string) (header.Header, error) {
	tm, err := makeHeaderTensorMap(tensors)
	if err != nil {
		return header.Header{}, err
	}
	return header.Header{
		Tensors:  tm,
		Metadata: metadata,
	}, nil
}

func makeHeaderTensorMap(tensors header.TensorSlice) (header.TensorMap, error) {
	tm := make(header.TensorMap, len(tensors))
	for _, t := range tensors {
		if _, ok := tm[t.Name]; ok {
			return nil, fmt.Errorf("duplicate tensor name %q", t.Name)
		}
		tm[t.Name] = t
	}
	return tm, nil
}

var headerPadding = [8]byte{' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '}

func writeHeader(w io.Writer, head header.Header) error {
	jsonHeader, err := head.MarshalJSON()
	if err != nil {
		return err
	}

	jsonLen := len(jsonHeader)
	// forcing 8-byte alignment
	toAlign := (8 - jsonLen%8) % 8

	if err = writeHeaderSize(w, jsonLen+toAlign); err != nil {
		return err
	}

	if _, err = w.Write(jsonHeader); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}
	if toAlign > 0 {
		if _, err = w.Write(headerPadding[:toAlign]); err != nil {
			return fmt.Errorf("failed to write header padding: %w", err)
		}
	}
	return nil
}

func writeHeaderSize(w io.Writer, n int) error {
	var arr [8]byte
	buf := arr[:]
	binary.LittleEndian.PutUint64(buf, uint64(n))
	if _, err := w.Write(buf); err != nil {
		return fmt.Errorf("failed to write header size: %w", err)
	}
	return nil
}

func writeTensors[T SerializableTensor](w io.Writer, tensors []T, tensorSlice header.TensorSlice) error {
	for i, t := range tensors {
		ht := tensorSlice[i]
		if err := writeTensor(w, t, ht); err != nil {
			return fmt.Errorf("failed to write data of tensor %q: %w", ht.Name, err)
		}
	}
	return nil
}

func writeTensor(w io.Writer, t io.WriterTo, ht header.Tensor) error {
	n, err := t.WriteTo(w)
	if err != nil {
		return err
	}

	expected := int64(ht.DataOffsets.End - ht.DataOffsets.Begin)
	if n != expected {
		return fmt.Errorf("expected %d written bytes, actual %d", expected, n)
	}
	return nil
}
