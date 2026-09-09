#pragma once
#include "AddressSupport.h"

namespace mlir::triton::laqs_packet {

using ScalarBindings = std::map<std::string, int64_t>;
struct IntegerRange { int64_t lo, hi; };

// Bindings describe the frozen launch, not inferred bounds on arbitrary calls.
// They are included in the compilation key and checked by FrozenLaunch.run.
class LaunchRanges {
public:
  explicit LaunchRanges(const ScalarBindings &bindings) : bindings(bindings) {}

  std::optional<IntegerRange> get(Value value, unsigned budget = 64) const {
    if (!budget) return {};
    auto integer = dyn_cast<IntegerType>(getElementTypeOrSelf(value.getType()));
    if (!integer || integer.getWidth() < 32 || integer.getWidth() > 64) return {};
    Attribute attr;
    if (matchPattern(value, m_Constant(&attr))) {
      std::optional<int64_t> n;
      if (auto scalar = dyn_cast<IntegerAttr>(attr)) n = scalar.getInt();
      if (auto tensor = dyn_cast<DenseIntElementsAttr>(attr); tensor && tensor.isSplat()) {
        n = tensor.getSplatValue<APInt>().getSExtValue();
      }
      if (n && *n >= INT32_MIN && *n <= INT32_MAX) return IntegerRange{*n, *n};
      return {};
    }
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (isa<FunctionOpInterface>(arg.getOwner()->getParentOp()))
        if (auto name = dyn_cast<NameLoc>(arg.getLoc())) {
          auto it = bindings.find(name.getName().str());
          if (it != bindings.end() && it->second >= INT32_MIN && it->second <= INT32_MAX)
            return IntegerRange{it->second, it->second};
        }
      return {};
    }
    Operation *op = value.getDefiningOp();
    if (!op) return {};
    if (auto pid = dyn_cast<GetProgramIdOp>(op)) {
      auto it = bindings.find("$grid" + std::to_string(static_cast<int>(pid.getAxis())));
      if (it != bindings.end() && it->second > 0 && it->second <= INT32_MAX)
        return IntegerRange{0, it->second - 1};
    }
    if (auto range = dyn_cast<MakeRangeOp>(op))
      return IntegerRange{range.getStart(), range.getEnd() - 1};
    StringRef name = op->getName().getStringRef();
    if (name == "tt.splat" || name == "tt.broadcast" || name == "tt.expand_dims" ||
        name == "tt.reshape" || name == "tt.trans" || name == "ttg.convert_layout" ||
        isa<arith::ExtSIOp>(op))
      return get(op->getOperand(0), budget - 1);
    if (isa<arith::ExtUIOp>(op)) {
      auto range = get(op->getOperand(0), budget - 1);
      return range && range->lo >= 0 ? range : std::nullopt;
    }
    if (op->getNumOperands() != 2) return {};
    auto a = get(op->getOperand(0), budget - 1), b = get(op->getOperand(1), budget - 1);
    if (!a || !b) return {};
    IntegerRange result;
    if (isa<arith::AddIOp>(op)) result = {a->lo + b->lo, a->hi + b->hi};
    else if (isa<arith::SubIOp>(op)) result = {a->lo - b->hi, a->hi - b->lo};
    else if (isa<arith::MulIOp>(op) && a->lo >= 0 && b->lo >= 0 &&
             a->hi <= INT32_MAX && b->hi <= INT32_MAX)
      result = {a->lo * b->lo, a->hi * b->hi};
    else if (isa<arith::DivSIOp, arith::DivUIOp>(op) && a->lo >= 0 && b->lo > 0)
      result = {a->lo / b->hi, a->hi / b->lo};
    else if (isa<arith::RemSIOp, arith::RemUIOp>(op) && a->lo >= 0 && b->lo == b->hi && b->lo > 0)
      result = {0, std::min(a->hi, b->lo - 1)};
    else if (isa<arith::MinSIOp>(op) || (isa<arith::MinUIOp>(op) && a->lo >= 0 && b->lo >= 0))
      result = {std::min(a->lo, b->lo), std::min(a->hi, b->hi)};
    else return {};
    // Range arithmetic must not silently assume that an i32 operation cannot wrap.
    if (result.lo < INT32_MIN || result.hi > INT32_MAX) return {};
    return result;
  }

  std::optional<int64_t> constant(Value value) const {
    auto range = get(value);
    if (range && range->lo == range->hi) return range->lo;
    return {};
  }

  // Exact conservative interval for the low bits, even when high coordinates
  // are unknown. Integer carries within this field remain represented.
  IntegerRange residue(Value value, unsigned bits, unsigned budget = 64) const {
    int64_t modulus = int64_t{1} << bits;
    auto integer = dyn_cast<IntegerType>(getElementTypeOrSelf(value.getType()));
    if (!integer || integer.getWidth() < bits) return {0, modulus - 1};
    auto reduce = [=](IntegerRange r) -> IntegerRange {
      if (r.lo >= 0 && r.lo / modulus == r.hi / modulus)
        return {r.lo % modulus, r.hi % modulus};
      return {0, modulus - 1};
    };
    if (auto range = get(value); range && range->lo >= 0 && range->lo / modulus == range->hi / modulus)
      return reduce(*range);
    Operation *op = value.getDefiningOp();
    if (!op || !budget) return {0, modulus - 1};
    StringRef name = op->getName().getStringRef();
    if (name == "tt.splat" || name == "tt.broadcast" || name == "tt.expand_dims" ||
        name == "tt.reshape" || name == "tt.trans" || name == "ttg.convert_layout" ||
        isa<arith::TruncIOp, arith::ExtSIOp, arith::ExtUIOp>(op))
      return residue(op->getOperand(0), bits, budget - 1);
    if (op->getNumOperands() != 2) return {0, modulus - 1};
    auto a = residue(op->getOperand(0), bits, budget - 1);
    auto b = residue(op->getOperand(1), bits, budget - 1);
    if (isa<arith::AddIOp>(op)) return reduce({a.lo + b.lo, a.hi + b.hi});
    if (isa<arith::MulIOp>(op)) return reduce({a.lo * b.lo, a.hi * b.hi});
    return {0, modulus - 1};
  }

private:
  const ScalarBindings &bindings;
};

struct AffineInitial { Value value; int64_t coefficient; };

// Recover x(iv) = x(lower) + coefficient * (iv - lower), modulo the low
// 32 bits used by packet addressing. Only integer arithmetic and tensor views
// can be hoisted; loop-carried values and nonlinear induction are rejected.
class LoopAffineInitial {
public:
  LoopAffineInitial(OpBuilder &builder, scf::ForOp loop, const LaunchRanges &ranges)
      : builder(builder), loop(loop), ranges(ranges) {}

  std::optional<AffineInitial> get(Value value, unsigned budget = 64) {
    if (!budget) return {};
    if (value == loop.getInductionVar()) return AffineInitial{loop.getLowerBound(), 1};
    if (loop.isDefinedOutsideOfLoop(value)) return AffineInitial{value, 0};
    if (auto it = cache.find(value); it != cache.end()) return it->second;
    auto integer = dyn_cast<IntegerType>(getElementTypeOrSelf(value.getType()));
    Operation *op = value.getDefiningOp();
    if (!integer || integer.getWidth() < 32 || integer.getWidth() > 64 || !op ||
        op->getNumResults() != 1 || op->getNumRegions()) return {};
    StringRef name = op->getName().getStringRef();
    bool view = name == "tt.splat" || name == "tt.broadcast" || name == "tt.expand_dims" ||
                name == "tt.reshape" || name == "tt.trans" || name == "ttg.convert_layout" ||
                isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(op);
    bool leaf = isa<arith::ConstantOp, MakeRangeOp, GetProgramIdOp>(op);
    if (!view && !leaf && !isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) return {};
    SmallVector<AffineInitial> operands;
    SmallVector<Value> initial;
    for (Value operand : op->getOperands()) {
      auto affine = get(operand, budget - 1);
      if (!affine) return {};
      operands.push_back(*affine);
      initial.push_back(affine->value);
    }
    int64_t coefficient = 0;
    if (view) coefficient = operands.front().coefficient;
    else if (!leaf) {
      auto a = operands[0], b = operands[1];
      if (isa<arith::AddIOp>(op)) coefficient = a.coefficient + b.coefficient;
      else if (isa<arith::SubIOp>(op)) coefficient = a.coefficient - b.coefficient;
      else if (a.coefficient || b.coefficient) {
        if (a.coefficient && b.coefficient) return {};
        auto factor = ranges.constant(a.coefficient ? b.value : a.value);
        if (!factor || *factor < 0 || *factor > INT32_MAX) return {};
        coefficient = *factor * (a.coefficient ? a.coefficient : b.coefficient);
      }
    }
    if (coefficient < 0 || coefficient > INT32_MAX) return {};
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(loop);
    OperationState state(op->getLoc(), op->getName());
    state.addOperands(initial);
    state.addTypes(op->getResultTypes());
    state.addAttributes(op->getAttrs());
    // Preserve properties too (notably arithmetic overflow flags).
    state.propertiesAttr = op->getPropertiesAsAttribute();
    auto result = AffineInitial{builder.create(state)->getResult(0), coefficient};
    cache[value] = result;
    return result;
  }

private:
  OpBuilder &builder;
  scf::ForOp loop;
  const LaunchRanges &ranges;
  llvm::DenseMap<Value, AffineInitial> cache;
};

// Prove F(x + t*step) = F(x) + t*delta in ordinary integer arithmetic.
// Every bit that can carry must map to a contiguous, increasing output field.
inline std::optional<int64_t> physicalIncrement(ArrayRef<uint64_t> rows,
    int64_t step, int64_t trips, Value initial, const LaunchRanges &ranges) {
  if (step <= 0 || !isOneHot(step) || trips < 0) return {};
  unsigned first = oneHotBit(step);
  if (first >= rows.size()) return {};
  SmallVector<unsigned> destination(rows.size());
  for (auto [i, row] : llvm::enumerate(rows)) destination[oneHotBit(row)] = i;
  unsigned end = first + 1;
  while (end < rows.size() && destination[end] == destination[first] + end - first) ++end;
  auto initialRange = ranges.residue(initial, end);
  int64_t room = (int64_t{1} << end) - 1 - initialRange.hi;
  if (std::max<int64_t>(0, trips - 1) > room / step) return {};
  return int64_t{1} << destination[first];
}

} // namespace mlir::triton::laqs_packet
