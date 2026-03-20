//===- Offload.h - Offloading binary format constants -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header contains type definitions and constants for the offloading binary
// format. These are stable interface types used across LLVM's offloading
// infrastructure.
//
// This is a header-only file requiring no library linkage, allowing lightweight
// usage by runtime libraries that need to understand the binary format without
// depending on the full Object library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BINARYFORMAT_OFFLOAD_H
#define LLVM_BINARYFORMAT_OFFLOAD_H

#include <cstdint>

namespace llvm {
namespace offload {

/// The producer of the associated offloading image.
enum OffloadKind : uint16_t {
  OFK_None = 0,
  OFK_OpenMP = (1 << 0),
  OFK_Cuda = (1 << 1),
  OFK_HIP = (1 << 2),
  OFK_SYCL = (1 << 3),
  OFK_LAST = (1 << 4),
};

/// The type of contents the offloading image contains.
enum ImageKind : uint16_t {
  IMG_None = 0,
  IMG_Object,
  IMG_Bitcode,
  IMG_Cubin,
  IMG_Fatbinary,
  IMG_PTX,
  IMG_SPIRV,
  IMG_LAST,
};

/// This is the record of an object that must be registered with the offloading
/// runtime.
struct EntryTy {
  /// Reserved bytes used to detect an older version of the struct, always zero.
  uint64_t Reserved = 0x0;
  /// The current version of the struct for runtime forward compatibility.
  uint16_t Version = 0x1;
  /// The expected consumer of this entry, e.g. SYCL or OpenMP.
  uint16_t Kind;
  /// Flags associated with the global.
  uint32_t Flags;
  /// The address of the global to be registered by the runtime.
  void *Address;
  /// The name of the symbol in the device image.
  char *SymbolName;
  /// The number of bytes the symbol takes.
  uint64_t Size;
  /// Extra generic data used to register this entry.
  uint64_t Data;
  /// An extra pointer, usually null.
  void *AuxAddr;
};

} // namespace offload
} // namespace llvm

#endif // LLVM_BINARYFORMAT_OFFLOAD_H
