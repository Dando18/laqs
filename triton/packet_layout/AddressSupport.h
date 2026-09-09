#pragma once
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Tools/PluginUtils.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace mlir::triton::laqs_packet {


struct LayoutSpec {
  unsigned argument = 0;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  std::vector<uint64_t> rows;
};

std::vector<StringRef> splitFields(StringRef text, char separator) {
  SmallVector<StringRef> fields;
  text.split(fields, separator, -1, false);
  return {fields.begin(), fields.end()};
}

std::vector<int64_t> parseSignedList(StringRef text, StringRef label) {
  std::vector<int64_t> values;
  for (StringRef field : splitFields(text, ',')) {
    int64_t value = 0;
    if (field.getAsInteger(10, value))
      llvm::report_fatal_error("invalid " + label + " in LAQS layout spec");
    values.push_back(value);
  }
  return values;
}

LayoutSpec parseSpec(StringRef text) {
  std::vector<StringRef> fields = splitFields(text, '|');
  if (fields.size() != 4)
    llvm::report_fatal_error(
        "LAQS layout spec must be argument|shape|strides|rows");

  LayoutSpec spec;
  if (fields[0].getAsInteger(10, spec.argument))
    llvm::report_fatal_error("invalid argument number in LAQS layout spec");
  spec.shape = parseSignedList(fields[1], "shape");
  spec.strides = parseSignedList(fields[2], "stride");
  for (int64_t row : parseSignedList(fields[3], "matrix row")) {
    if (row < 0)
      llvm::report_fatal_error("LAQS matrix rows must be nonnegative");
    spec.rows.push_back(static_cast<uint64_t>(row));
  }
  if (spec.shape.empty() || spec.shape.size() != spec.strides.size())
    llvm::report_fatal_error("LAQS shape and stride ranks do not match");
  if (spec.rows.empty() || spec.rows.size() > 62)
    llvm::report_fatal_error("LAQS layouts require between 1 and 62 bits");
  for (int64_t extent : spec.shape) {
    if (extent <= 0)
      llvm::report_fatal_error("LAQS layout extents must be positive");
  }
  for (int64_t stride : spec.strides) {
    if (stride <= 0)
      llvm::report_fatal_error("LAQS layout strides must be positive");
  }
  return spec;
}

bool preservesPointerProvenance(StringRef name) {
  return name == "tt.bitcast" || name == "tt.splat" || name == "tt.broadcast" ||
         name == "tt.expand_dims" || name == "tt.reshape" ||
         name == "tt.trans" || name == "ttg.convert_layout" ||
         name == "builtin.unrealized_conversion_cast";
}

void collectBaseArguments(Value value, llvm::SetVector<unsigned> &bases,
                          llvm::SmallPtrSetImpl<void *> *active = nullptr) {
  llvm::SmallPtrSet<void *, 16> ownedActive;
  if (!active)
    active = &ownedActive;
  void *key = value.getAsOpaquePointer();
  if (!active->insert(key).second)
    return;

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Block *block = argument.getOwner();
    Operation *parent = block->getParentOp();
    if (isa<FunctionOpInterface>(parent)) {
      bases.insert(argument.getArgNumber());
    } else if (auto loop = dyn_cast<scf::ForOp>(parent)) {
      if (argument != loop.getInductionVar()) {
        unsigned slot = argument.getArgNumber() - 1;
        collectBaseArguments(loop.getInitArgs()[slot], bases, active);
        collectBaseArguments(loop.getBody()->getTerminator()->getOperand(slot),
                             bases, active);
      }
    }
    active->erase(key);
    return;
  }

  Operation *definition = value.getDefiningOp();
  if (!definition) {
    active->erase(key);
    return;
  }
  StringRef name = definition->getName().getStringRef();
  if (name == "tt.addptr" || preservesPointerProvenance(name)) {
    collectBaseArguments(definition->getOperand(0), bases, active);
  } else if (name == "arith.select") {
    collectBaseArguments(definition->getOperand(1), bases, active);
    collectBaseArguments(definition->getOperand(2), bases, active);
  } else if (auto conditional = dyn_cast<scf::IfOp>(definition)) {
    unsigned result = cast<OpResult>(value).getResultNumber();
    collectBaseArguments(conditional.thenYield()->getOperand(result), bases,
                         active);
    collectBaseArguments(conditional.elseYield()->getOperand(result), bases,
                         active);
  } else if (auto loop = dyn_cast<scf::ForOp>(definition)) {
    unsigned result = cast<OpResult>(value).getResultNumber();
    collectBaseArguments(loop.getInitArgs()[result], bases, active);
    collectBaseArguments(loop.getBody()->getTerminator()->getOperand(result),
                         bases, active);
  }
  active->erase(key);
}

Type offsetType(OpBuilder &builder, Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return RankedTensorType::get(tensor.getShape(), builder.getI32Type(),
                                 tensor.getEncoding());
  return builder.getI32Type();
}

Value constantLike(OpBuilder &builder, Location location, Type type,
                   uint64_t value) {
  Type scalarType = type;
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    scalarType = tensor.getElementType();
  auto scalar = builder.getIntegerAttr(scalarType, value);
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return arith::ConstantOp::create(builder, location,
                                     SplatElementsAttr::get(tensor, scalar));
  return arith::ConstantOp::create(builder, location, scalar);
}

unsigned modeBits(int64_t extent) {
  unsigned bits = 0;
  for (uint64_t capacity = 1; capacity < static_cast<uint64_t>(extent);
       capacity <<= 1)
    ++bits;
  return bits;
}

bool isOneHot(uint64_t value) { return value && !(value & (value - 1)); }

unsigned oneHotBit(uint64_t value) {
  unsigned bit = 0;
  while (value > 1) {
    value >>= 1;
    ++bit;
  }
  return bit;
}

uint64_t bitMask(unsigned start, unsigned width) {
  return ((uint64_t{1} << width) - 1) << start;
}

std::optional<std::vector<uint64_t>>
rowsForDensePowerOfTwoOffset(const LayoutSpec &spec) {
  std::vector<unsigned> elementShifts(spec.shape.size());
  uint64_t expectedStride = 1;
  for (unsigned reverse = 0; reverse < spec.shape.size(); ++reverse) {
    unsigned dimension = spec.shape.size() - reverse - 1;
    uint64_t extent = static_cast<uint64_t>(spec.shape[dimension]);
    if (!isOneHot(extent) ||
        static_cast<uint64_t>(spec.strides[dimension]) != expectedStride)
      return std::nullopt;
    elementShifts[dimension] = oneHotBit(expectedStride);
    if (expectedStride > std::numeric_limits<uint64_t>::max() / extent)
      return std::nullopt;
    expectedStride *= extent;
  }

  std::vector<unsigned> logicalShifts(spec.shape.size());
  unsigned logicalShift = 0;
  for (unsigned dimension = 0; dimension < spec.shape.size(); ++dimension) {
    logicalShifts[dimension] = logicalShift;
    logicalShift += modeBits(spec.shape[dimension]);
  }
  if (logicalShift != spec.rows.size())
    llvm::report_fatal_error("LAQS row count does not match shape envelope");

  std::vector<uint64_t> rows(spec.rows.size(), 0);
  for (unsigned physicalBit = 0; physicalBit < spec.rows.size();
       ++physicalBit) {
    uint64_t sourceRow = spec.rows[physicalBit];
    for (unsigned dimension = 0; dimension < spec.shape.size(); ++dimension) {
      unsigned width = modeBits(spec.shape[dimension]);
      for (unsigned bit = 0; bit < width; ++bit) {
        if (sourceRow & (uint64_t{1} << (logicalShifts[dimension] + bit)))
          rows[physicalBit] |= uint64_t{1} << (elementShifts[dimension] + bit);
      }
    }
  }
  return rows;
}

Value moveBits(OpBuilder &builder, Location location, Value logical,
               unsigned sourceStart, unsigned physicalStart, unsigned width) {
  Type type = logical.getType();
  Value result = arith::AndIOp::create(
      builder, location, logical,
      constantLike(builder, location, type, bitMask(sourceStart, width)));
  if (physicalStart > sourceStart) {
    result = arith::ShLIOp::create(
        builder, location, result,
        constantLike(builder, location, type, physicalStart - sourceStart));
  } else if (sourceStart > physicalStart) {
    result = arith::ShRUIOp::create(
        builder, location, result,
        constantLike(builder, location, type, sourceStart - physicalStart));
  }
  return result;
}


} // namespace mlir::triton::laqs_packet
