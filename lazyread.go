// Copyright 2023 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetensors

import (
	"errors"
	"fmt"
	"io"
	"math"
	"math/bits"

	"github.com/nlpodyssey/safetensors/dtype"
	"github.com/nlpodyssey/safetensors/header"
)

// LazyST (short for "LazySafeTensors") allows to read safetensors content
// lazy-loading data of individual tensors.
type LazyST struct {
	rs       io.ReadSeeker
	tensors  header.TensorMap
	metadata header.Metadata
	// dataOffset is the byte-buffer offset relative to the start of rs
	dataOffset int64
}

// LazyTensor provides information about a tensor and allows lazy loading
// its data.
//
// It only retains in-memory information from the safetensors header, and only
// small references to know how to retrieve the tensor's data later
// (lazy loading).
type LazyTensor struct {
	rs io.ReadSeeker
	t  header.Tensor
	// dataOffset is the byte-buffer offset relative to the start of rs
	dataOffset int64
}

// NewLazy reads from "rs" the safetensors header and validates it, then
// returns a new LazyST in case of success, otherwise nil and an error.
//
// If headerSizeLimit is set to a positive number, its value is used to
// limit the reading of safetensors header. This can be useful to guard
// against attacks or tampered/garbage data, avoiding giant memory allocations
// to hold header information. A value of zero, or a negative number, have
// no limiting effects.
//
// The current "seek" position of "rs" is used as a base for all further
// seek-based operations to read tensor data.
//
// In order to allow lazy loading of tensors data, the given io.ReadSeeker
// must remain available for operations as long as you are handling
// a LazyST object and any LazyTensor obtained from it. For example,
// if the given "rs" is a file, it should not be closed until you got all
// the tensor data you need.
//
// Conversely, a fully loaded Tensor is completely independent, and has no
// bonds with any LazyST, LazyTensor or io.ReadSeeker element.
func NewLazy(rs io.ReadSeeker, headerSizeLimit int) (*LazyST, error) {
	initialOffset, err := rs.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, fmt.Errorf("failed to get initial offset: %w", err)
	}

	head, err := readValidHeader(rs, headerSizeLimit)
	if err != nil {
		return nil, err
	}

	byteBufferOffset, err := checkedAddNonNegInt64(initialOffset, int64(head.ByteBufferOffset))
	if err != nil {
		return nil, fmt.Errorf("failed to calculate total byte-buffer offset: %w", err)
	}

	return &LazyST{
		rs:         rs,
		tensors:    head.Tensors,
		metadata:   head.Metadata,
		dataOffset: byteBufferOffset,
	}, nil
}

// Metadata returns the free-form key/value string pairs as read from the
// safetensors header. It can be nil.
//
// It returns the same value retained internally, without copy.
func (st *LazyST) Metadata() map[string]string {
	return st.metadata
}

// TensorNames returns a slice of the names of all tensors.
//
// If there are no tensors it returns nil, otherwise a new slice of
// strings is allocated and returned.
func (st *LazyST) TensorNames() []string {
	if len(st.tensors) == 0 {
		return nil
	}
	names := make([]string, 0, len(st.tensors))
	for name := range st.tensors {
		names = append(names, name)
	}
	return names
}

// AllTensors reads, converts, and loads in memory the data of all available
// tensors, returning a list of fully resolved Tensor objects.
//
// This can be used as alternative to asking individual LazyTensors
// and then converting them to Tensors. It's useful in case you need all
// tensor loaded in memory at once, and it also has the small advantage of
// having slightly smaller overhead (because of fewer "seek" calls).
func (st *LazyST) AllTensors() ([]Tensor, error) {
	if _, err := st.rs.Seek(st.dataOffset, io.SeekStart); err != nil {
		return nil, fmt.Errorf("failed to seek to byte-buffer offset: %w", err)
	}
	return readAllTensors(st.tensors, st.rs, true)
}

// AllRawTensors is similar to LazyST.AllTensors, but returns RawTensor objects.
func (st *LazyST) AllRawTensors() ([]RawTensor, error) {
	if _, err := st.rs.Seek(st.dataOffset, io.SeekStart); err != nil {
		return nil, fmt.Errorf("failed to seek to byte-buffer offset: %w", err)
	}
	return readAllRawTensors(st.tensors, st.rs, true)
}

// LazyTensor returns a LazyTensor by its name, and whether it has been found.
//
// If ok is false, the LazyTensor is the zero-value, and must not be used.
func (st *LazyST) LazyTensor(name string) (_ LazyTensor, ok bool) {
	t, ok := st.tensors[name]
	if !ok {
		return LazyTensor{}, false
	}
	return LazyTensor{
		rs:         st.rs,
		t:          t,
		dataOffset: st.dataOffset,
	}, true
}

// Name returns the name of the tensor.
func (lt LazyTensor) Name() string {
	return lt.t.Name
}

// DType returns the safetensors-specific data type of the tensor.
func (lt LazyTensor) DType() dtype.DType {
	return lt.t.DType
}

// Shape returns the shape of the tensor.
//
// If the shape is zero-length, it returns nil, otherwise a new slice
// is allocated and returned (the shape is copied to prevent tampering).
func (lt LazyTensor) Shape() []int {
	return copyShape(lt.t.Shape)
}

// Tensor converts the LazyTensor into a new full Tensor, with data read,
// interpreted and loaded in memory.
//
// Data is interpreted converting raw []byte content to a specifically
// typed slice.
//
// If a failure happens at any stage, an error is returned, while the Tensor
// is the zero-value, and must not be used.
func (lt LazyTensor) Tensor() (Tensor, error) {
	if err := lt.seekTensorData(); err != nil {
		return Tensor{}, err
	}
	return readTensor(lt.t, lt.rs, true)
}

// RawTensor converts the LazyTensor into a RawTensor, with raw []byte
// data read and loaded into memory.
//
// Unlike LazyTensor.Tensor function, data is returned as it is, without being
// interpreted and converted to a specifically typed slice.
//
// If a failure happens at any stage, an error is returned, while the RawTensor
// is the zero-value, and must not be used.
func (lt LazyTensor) RawTensor() (RawTensor, error) {
	data, err := lt.ReadData()
	if err != nil {
		return RawTensor{}, err
	}
	return RawTensor{
		name:  lt.t.Name,
		dType: lt.t.DType,
		shape: copyShape(lt.t.Shape),
		data:  data,
	}, nil
}

// ReadData reads and returns the raw []byte data of the tensor.
//
// Safetensors data is expected to be little-endian and row-major ("C")
// ordered. There is no striding.
func (lt LazyTensor) ReadData() ([]byte, error) {
	size := lt.t.DataOffsets.End - lt.t.DataOffsets.Begin
	if size == 0 {
		return nil, nil
	}
	if err := lt.seekTensorData(); err != nil {
		return nil, err
	}
	data := make([]byte, size)
	if _, err := io.ReadFull(lt.rs, data); err != nil {
		return nil, fmt.Errorf("failed to read tensor data: %w", err)
	}
	return data, nil
}

// WriteTo reads raw tensor data and copies it to the given io.Writer.
// This method satisfies io.WriterTo interface.
//
// Data is copied with io.CopyN, so, apart from an internal buffer, this
// function does not allocate the entire tensor's data in memory.
func (lt LazyTensor) WriteTo(w io.Writer) (int64, error) {
	size := lt.t.DataOffsets.End - lt.t.DataOffsets.Begin
	if size == 0 {
		return 0, nil
	}
	if err := lt.seekTensorData(); err != nil {
		return 0, err
	}
	return io.CopyN(w, lt.rs, int64(size))
}

func (lt LazyTensor) seekTensorData() error {
	offset, err := checkedAddNonNegInt64(lt.dataOffset, int64(lt.t.DataOffsets.Begin))
	if err != nil {
		return fmt.Errorf("failed to calculate tensor data offset: %w", err)
	}
	if _, err = lt.rs.Seek(offset, io.SeekStart); err != nil {
		return fmt.Errorf("failed to seek to tensor data offset: %w", err)
	}
	return nil
}

var errInt64SumOverflow = errors.New("int64 sum overflow")

func checkedAddNonNegInt64(a, b int64) (int64, error) {
	if a < 0 || b < 0 {
		return 0, fmt.Errorf("unexpected negative number")
	}
	if a == 0 || b == 0 {
		return a + b, nil
	}
	sum, carry := bits.Add64(uint64(a), uint64(b), 0)
	if carry != 0 || sum > math.MaxInt64 {
		return 0, errInt64SumOverflow
	}
	return int64(sum), nil
}
