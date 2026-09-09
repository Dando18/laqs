#include "AddressSupport.h"
#include "RecurrenceSupport.h"
#include "llvm/Support/JSON.h"

namespace mlir::triton::laqs_packet {
namespace {

Value physicalOffset(OpBuilder &builder, Value logical, const LayoutSpec &spec, unsigned packetBits = 0);

// Recover integer SSA offsets, preserving the original integer arithmetic.
// The admitted permutation reads fewer than 31 low bits. Modulo-2^32 offset
// state therefore preserves every used bit, including through integer carries.
class OffsetBuilder {
public:
  OffsetBuilder(MLIRContext *context, const ScalarBindings &bindings, bool recurrent)
      : builder(context), ranges(bindings), recurrent(recurrent) {}

  Value physical(Value pointer, const LayoutSpec &spec, unsigned packetBits) {
    if (auto it = physicals.find(pointer); it != physicals.end()) return it->second;
    if (auto op = pointer.getDefiningOp(); op && preservesPointerProvenance(op->getName().getStringRef()) &&
        !isa<BitcastOp>(op)) {
      if (auto it = physicals.find(op->getOperand(0)); it != physicals.end()) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(op);
        OperationState state(op->getLoc(), op->getName());
        state.addOperands(it->second);
        state.addTypes(offsetType(builder, pointer.getType()));
        state.addAttributes(op->getAttrs());
        for (auto name : {"tt.contiguity", "tt.divisibility", "tt.constancy"}) state.attributes.erase(name);
        return builder.create(state)->getResult(0);
      }
    }
    Value logical = get(pointer);
    if (!logical) return {};
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfterValue(logical);
    return physicalOffset(builder, logical, spec, packetBits);
  }

  unsigned recurrenceCount() const { return recurrence_count; }

  Value get(Value value) {
    if (auto it = offsets.find(value); it != offsets.end())
      return it->second;
    OpBuilder::InsertionGuard guard(builder);
    Location loc = value.getLoc();
    Type type = offsetType(builder, value.getType());
    Value result;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (!isa<FunctionOpInterface>(arg.getOwner()->getParentOp()))
        return {};
      builder.setInsertionPointToStart(arg.getOwner());
      result = constantLike(builder, loc, type, 0);
    } else {
      Operation *op = value.getDefiningOp();
      if (!op)
        return {};
      builder.setInsertionPointAfter(op);
      if (auto add = dyn_cast<AddPtrOp>(op)) {
        Value left = get(add.getPtr());
        if (!left)
          return {};
        Value right = lowOffset(add.getOffset());
        if (!right)
          return {};
        result = arith::AddIOp::create(builder, loc, left, right);
      } else if (preservesPointerProvenance(op->getName().getStringRef())) {
        Value operand = get(op->getOperand(0));
        if (!operand)
          return {};
        if (isa<BitcastOp>(op)) {
          auto pointeeBytes = [](Type type) {
            if (auto tensor = dyn_cast<RankedTensorType>(type))
              type = tensor.getElementType();
            return (cast<PointerType>(type).getPointeeType().getIntOrFloatBitWidth() + 7) / 8;
          };
          if (pointeeBytes(value.getType()) != pointeeBytes(op->getOperand(0).getType()))
            return {};
          result = operand;
        } else if (op->getName().getStringRef() == "builtin.unrealized_conversion_cast") {
          return {};
        } else {
          OperationState state(loc, op->getName());
          state.addOperands(operand);
          state.addTypes(type);
          state.addAttributes(op->getAttrs());
          state.attributes.erase("tt.contiguity");
          state.attributes.erase("tt.divisibility");
          state.attributes.erase("tt.constancy");
          result = builder.create(state)->getResult(0);
        }
      } else if (auto select = dyn_cast<arith::SelectOp>(op)) {
        Value yes = get(select.getTrueValue()), no = get(select.getFalseValue());
        if (!yes || !no)
          return {};
        result = arith::SelectOp::create(builder, loc, select.getCondition(), yes, no);
      } else {
        return {};
      }
    }
    offsets[value] = result;
    return result;
  }

  LogicalResult extendLoop(scf::ForOp loop,
                           const std::map<unsigned, LayoutSpec> &specs) {
    SmallVector<unsigned> slots;
    SmallVector<int64_t> increments;
    SmallVector<Value> initial(loop.getInitArgs());
    for (auto [slot, value] : llvm::enumerate(loop.getInitArgs())) {
      llvm::SetVector<unsigned> bases;
      collectBaseArguments(value, bases);
      if (bases.size() != 1 || !specs.count(bases.front()))
        continue;
      Value offset = get(value);
      if (!offset)
        return loop.emitError("packet layout cannot recover loop initial offset");
      if (recurrent) {
        auto update = loop.getBody()->getTerminator()->getOperand(slot).getDefiningOp<AddPtrOp>();
        auto lower = ranges.constant(loop.getLowerBound()), upper = ranges.constant(loop.getUpperBound());
        auto loopStep = ranges.constant(loop.getStep());
        if (!update || update.getPtr() != loop.getRegionIterArg(slot) ||
            !lower || !upper || !loopStep || *loopStep <= 0 ||
            !loop.getResult(slot).use_empty() ||
            !llvm::all_of(update->getUsers(), [&](Operation *user) { return user == loop.getBody()->getTerminator(); }))
          return loop.emitError("unsupported physical pointer recurrence: require a bounded constant update and no escaping final pointer");
        auto stride = ranges.constant(update.getOffset());
        int64_t trips = *upper <= *lower ? 0 : (*upper - *lower + *loopStep - 1) / *loopStep;
        auto delta = stride ? physicalIncrement(*rowsForDensePowerOfTwoOffset(specs.at(bases.front())),
                                                *stride, trips, offset, ranges) : std::nullopt;
        if (!delta) return loop.emitError("physical pointer recurrence proof failed: unsupported stride or coordinate carry");
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(loop);
        offset = physicalOffset(builder, offset, specs.at(bases.front()));
        increments.push_back(*delta);
      }
      slots.push_back(slot);
      initial.push_back(offset);
    }
    SmallVector<Value> affinePointers;
    if (recurrent) {
      auto lower = ranges.constant(loop.getLowerBound()), upper = ranges.constant(loop.getUpperBound());
      auto loopStep = ranges.constant(loop.getStep());
      if (lower && upper && loopStep && *loopStep > 0) {
        int64_t trips = *upper <= *lower ? 0 : (*upper - *lower + *loopStep - 1) / *loopStep;
        LoopAffineInitial affine(builder, loop, ranges);
        loop.walk([&](LoadOp load) {
          if (load->getParentOfType<scf::ForOp>() != loop) return;
          Value pointer = load.getPtr();
          if (llvm::is_contained(affinePointers, pointer)) return;
          llvm::SetVector<unsigned> bases;
          collectBaseArguments(pointer, bases);
          if (bases.size() != 1 || !specs.count(bases.front())) return;
          Value logical = get(pointer);
          if (!logical) return;
          auto start = affine.get(logical);
          if (!start || !start->coefficient || start->coefficient > INT32_MAX / *loopStep) return;
          const auto &spec = specs.at(bases.front());
          auto delta = physicalIncrement(*rowsForDensePowerOfTwoOffset(spec),
              start->coefficient * *loopStep, trips, start->value, ranges);
          if (!delta) return;
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(loop);
          initial.push_back(physicalOffset(builder, start->value, spec));
          increments.push_back(*delta);
          affinePointers.push_back(pointer);
        });
      }
    }
    if (slots.empty() && affinePointers.empty())
      return success();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(loop);
    auto replacement = scf::ForOp::create(builder, loop.getLoc(), loop.getLowerBound(),
        loop.getUpperBound(), loop.getStep(), initial);
    replacement->setAttrs(loop->getAttrs());
    Block *body = replacement.getBody();
    // The default builder inserts a terminator only for loops without results.
    if (!body->empty())
      body->back().erase();
    unsigned originalCount = loop.getInitArgs().size();
    loop.getInductionVar().replaceAllUsesWith(replacement.getInductionVar());
    for (unsigned i = 0; i < originalCount; ++i)
      loop.getRegionIterArg(i).replaceAllUsesWith(replacement.getRegionIterArg(i));
    for (auto [i, slot] : llvm::enumerate(slots)) {
      auto &map = recurrent ? physicals : offsets;
      map[replacement.getRegionIterArg(slot)] = replacement.getRegionIterArg(originalCount + i);
    }
    for (auto [i, pointer] : llvm::enumerate(affinePointers))
      physicals[pointer] = replacement.getRegionIterArg(originalCount + slots.size() + i);
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    SmallVector<Value> yielded(yield.getOperands());
    while (&loop.getBody()->front() != yield.getOperation())
      loop.getBody()->front().moveBefore(body, body->end());
    builder.setInsertionPointToEnd(body);
    for (auto [i, slot] : llvm::enumerate(slots)) {
      Value offset;
      if (recurrent) {
        Value previous = replacement.getRegionIterArg(originalCount + i);
        offset = arith::AddIOp::create(builder, loop.getLoc(), previous,
            constantLike(builder, loop.getLoc(), previous.getType(), increments[i]),
            arith::IntegerOverflowFlags::nsw | arith::IntegerOverflowFlags::nuw);
        // Every used offset is in the <31-bit envelope. Even the unused
        // final update adds at most 2^29, so this i32 induction cannot wrap.
        ++recurrence_count;
      } else offset = get(yielded[slot]);
      if (!offset)
        return replacement.emitError("packet layout cannot recover loop yielded offset");
      yielded.push_back(offset);
    }
    for (unsigned i = slots.size(); i < increments.size(); ++i) {
      Value previous = replacement.getRegionIterArg(originalCount + i);
      yielded.push_back(arith::AddIOp::create(builder, loop.getLoc(), previous,
          constantLike(builder, loop.getLoc(), previous.getType(), increments[i]),
          arith::IntegerOverflowFlags::nsw | arith::IntegerOverflowFlags::nuw));
      ++recurrence_count;
    }
    builder.setInsertionPointToEnd(body);
    scf::YieldOp::create(builder, loop.getLoc(), yielded);
    for (auto [i, slot] : llvm::enumerate(slots)) {
      auto &map = recurrent ? physicals : offsets;
      map[replacement.getResult(slot)] = replacement.getResult(originalCount + i);
    }
    for (unsigned i = 0; i < originalCount; ++i)
      loop.getResult(i).replaceAllUsesWith(replacement.getResult(i));
    loop.erase();
    return success();
  }

private:
  Value lowOffset(Value value) {
    Type type = value.getType();
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      type = tensor.getElementType();
    auto integer = dyn_cast<IntegerType>(type);
    if (!integer || integer.getWidth() > 64)
      return {};
    if (integer.getWidth() == 32)
      return value;
    if (integer.getWidth() > 32)
      return arith::TruncIOp::create(builder, value.getLoc(),
          offsetType(builder, value.getType()), value);
    return arith::ExtSIOp::create(builder, value.getLoc(),
        offsetType(builder, value.getType()), value);
  }
  OpBuilder builder;
  llvm::DenseMap<Value, Value> offsets;
  llvm::DenseMap<Value, Value> physicals;
  LaunchRanges ranges;
  bool recurrent;
  unsigned recurrence_count = 0;
};

std::optional<uint64_t> constant(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return std::nullopt;
  if (auto scalar = dyn_cast<IntegerAttr>(attr))
    return scalar.getValue().getZExtValue();
  if (auto elements = dyn_cast<DenseIntElementsAttr>(attr); elements && elements.isSplat())
    return elements.getSplatValue<APInt>().getZExtValue();
  return std::nullopt;
}

bool shapeOnly(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "tt.splat" || name == "tt.broadcast" || name == "tt.expand_dims" ||
         name == "tt.reshape" || name == "tt.trans" || name == "ttg.convert_layout";
}

// A conservative possible-one-bit mask. We distribute over integer addition
// only when the operands occupy disjoint fields, so no carry is possible.
uint64_t possibleBits(Value value, unsigned budget = 32) {
  if (!budget)
    return ~uint64_t{0};
  if (auto c = constant(value))
    return *c;
  Operation *op = value.getDefiningOp();
  if (!op)
    return ~uint64_t{0};
  if (auto range = dyn_cast<MakeRangeOp>(op))
    return range.getStart() >= 0 ? bitMask(0, modeBits(range.getEnd())) : ~uint64_t{0};
  if (shapeOnly(op))
    return possibleBits(op->getOperand(0), budget - 1);
  if (isa<arith::ExtUIOp>(op))
    return possibleBits(op->getOperand(0), budget - 1);
  if (isa<arith::ExtSIOp>(op)) {
    uint64_t bits = possibleBits(op->getOperand(0), budget - 1);
    Type type = op->getOperand(0).getType();
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      type = tensor.getElementType();
    unsigned width = cast<IntegerType>(type).getWidth();
    return bits & (uint64_t{1} << (width - 1)) ? ~uint64_t{0} : bits;
  }
  if (op->getNumOperands() == 2) {
    uint64_t left = possibleBits(op->getOperand(0), budget - 1);
    uint64_t right = possibleBits(op->getOperand(1), budget - 1);
    if (isa<arith::AndIOp>(op))
      return left & right;
    if (isa<arith::OrIOp, arith::XOrIOp>(op))
      return left | right;
    if (isa<arith::AddIOp>(op) && !(left & right))
      return left | right;
    if (isa<arith::ShLIOp>(op))
      if (auto amount = constant(op->getOperand(1)); amount && *amount < 64)
        return left << *amount;
    if (isa<arith::MulIOp>(op))
      if (auto factor = constant(op->getOperand(1)); factor && isOneHot(*factor))
        return left << oneHotBit(*factor);
  }
  return ~uint64_t{0};
}

class PacketAddressBuilder {
public:
  explicit PacketAddressBuilder(OpBuilder &builder) : b(builder) {}

  Value project(Value value, unsigned start, unsigned width, unsigned budget = 32) {
    Location loc = value.getLoc();
    Type type = value.getType();
    uint64_t mask = bitMask(start, width);
    if (!(possibleBits(value) & mask))
      return constantLike(b, loc, type, 0);
    if (auto c = constant(value))
      return constantLike(b, loc, type, (*c & mask) >> start);
    Operation *op = value.getDefiningOp();
    if (op && budget) {
      if (shapeOnly(op)) {
        Value operand = project(op->getOperand(0), start, width, budget - 1);
        OperationState state(loc, op->getName());
        state.addOperands(operand);
        state.addTypes(type);
        state.addAttributes(op->getAttrs());
        state.attributes.erase("tt.contiguity");
        state.attributes.erase("tt.divisibility");
        state.attributes.erase("tt.constancy");
        return b.create(state)->getResult(0);
      }
      if (isa<arith::AddIOp>(op) &&
          !(possibleBits(op->getOperand(0)) & possibleBits(op->getOperand(1)))) {
        return arith::AddIOp::create(b, loc,
            project(op->getOperand(0), start, width, budget - 1),
            project(op->getOperand(1), start, width, budget - 1));
      }
      if (isa<arith::ExtSIOp, arith::ExtUIOp>(op)) {
        Type inner = op->getOperand(0).getType();
        if (auto tensor = dyn_cast<RankedTensorType>(inner))
          inner = tensor.getElementType();
        if (start + width < cast<IntegerType>(inner).getWidth()) {
          Value field = project(op->getOperand(0), start, width, budget - 1);
          return arith::ExtUIOp::create(b, loc, type, field);
        }
      }
    }
    return moveBits(b, loc, value, start, 0, width);
  }

  Value build(Value logical, ArrayRef<uint64_t> rows, unsigned packetBits) {
    Location loc = logical.getLoc();
    Type type = logical.getType();
    Value result = packetBits ? project(logical, 0, packetBits)
                             : constantLike(b, loc, type, 0);
    for (unsigned target = packetBits; target < rows.size();) {
      unsigned source = oneHotBit(rows[target]), width = 1;
      while (target + width < rows.size() && rows[target + width] == (uint64_t{1} << (source + width)))
        ++width;
      Value field = project(logical, source, width);
      if (target)
        field = arith::ShLIOp::create(b, loc, field, constantLike(b, loc, type, target));
      result = arith::AddIOp::create(b, loc, result, field);
      target += width;
    }
    return result;
  }
private:
  OpBuilder &b;
};

Value physicalOffset(OpBuilder &builder, Value logical, const LayoutSpec &spec, unsigned packetBits) {
  return PacketAddressBuilder(builder).build(logical, *rowsForDensePowerOfTwoOffset(spec), packetBits);
}

struct SiteContract {
  unsigned argument, packet, alignment, maskAlignment;
  SmallVector<int64_t> contiguity, divisibility, constancy;
};

class PacketLayoutPass : public PassWrapper<PacketLayoutPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PacketLayoutPass)
  PacketLayoutPass() = default;
  explicit PacketLayoutPass(std::vector<LayoutSpec> specs, bool inspect = false,
      std::string mode = "repaired", ScalarBindings bindings = {})
      : specs(std::move(specs)), inspect(inspect), mode(std::move(mode)), bindings(std::move(bindings)) {}
  StringRef getArgument() const override { return "laqs-packet-layout"; }
  StringRef getDescription() const override { return "materialize canonical packet bases from integer SSA offsets"; }

  void runOnOperation() override {
    std::map<unsigned, LayoutSpec> byArgument;
    for (const auto &spec : specs) {
      auto rows = rowsForDensePowerOfTwoOffset(spec);
      uint64_t seen = 0;
      if (!rows || spec.rows.size() >= 31) {
        getOperation().emitError("packet templates require dense power-of-two shapes with fewer than 31 address bits");
        return signalPassFailure();
      }
      for (uint64_t row : *rows) {
        if (!isOneHot(row) || (row & seen)) {
          getOperation().emitError("packet templates require a bit permutation");
          return signalPassFailure();
        }
        seen |= row;
      }
      if (!byArgument.emplace(spec.argument, spec).second) {
        getOperation().emitError("duplicate packet argument");
        return signalPassFailure();
      }
    }
    ModuleAxisInfoAnalysis native(getOperation());
    llvm::DenseMap<Operation *, SiteContract> contracts;
    bool invalid = false;
    getOperation().walk([&](Operation *op) {
      // Fail closed before a pointer can escape into an untracked operation.
      bool supportedUse = isa<AddPtrOp, LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp,
                              MakeTensorDescOp, arith::SelectOp, scf::ForOp, scf::YieldOp>(op) ||
                          preservesPointerProvenance(op->getName().getStringRef());
      if (!supportedUse) {
        for (Value operand : op->getOperands()) {
          Type type = operand.getType();
          if (auto tensor = dyn_cast<RankedTensorType>(type)) type = tensor.getElementType();
          if (!isa<PointerType>(type)) continue;
          llvm::SetVector<unsigned> bases;
          collectBaseArguments(operand, bases);
          if (llvm::any_of(bases, [&](unsigned arg) { return byArgument.count(arg); })) {
            op->emitError("selected storage escapes supported integer-offset provenance");
            invalid = true;
          }
        }
      }
      Value ptr;
      if (auto load = dyn_cast<LoadOp>(op)) ptr = load.getPtr();
      else if (auto store = dyn_cast<StoreOp>(op)) ptr = store.getPtr();
      else if (auto atomic = dyn_cast<AtomicRMWOp>(op)) ptr = atomic.getPtr();
      else if (auto atomic = dyn_cast<AtomicCASOp>(op)) ptr = atomic.getPtr();
      else if (auto descriptor = dyn_cast<MakeTensorDescOp>(op)) ptr = descriptor.getBase();
      else return;
      llvm::SetVector<unsigned> bases;
      collectBaseArguments(ptr, bases);
      bool selected = llvm::any_of(bases, [&](unsigned arg) { return byArgument.count(arg); });
      if (!selected) return;
      if (bases.size() != 1 || !isa<LoadOp>(op)) {
        op->emitError("packet templates require unambiguous loads from read-only storage; descriptors, stores and atomics are unsupported");
        invalid = true;
        return;
      }
      auto load = cast<LoadOp>(op);
      Type type = ptr.getType();
      if (auto tensor = dyn_cast<RankedTensorType>(type)) type = tensor.getElementType();
      unsigned bytes = (cast<PointerType>(type).getPointeeType().getIntOrFloatBitWidth() + 7) / 8;
      unsigned packet = std::min(16u / bytes, native.getContiguity(ptr));
      unsigned mask = load.getMask() ? native.getMaskAlignment(load.getMask()) : packet;
      packet = std::max(1u, std::min(packet, mask));
      auto relative = *rowsForDensePowerOfTwoOffset(byArgument.at(bases.front()));
      unsigned bits = modeBits(packet);
      for (unsigned i = 0; i < bits; ++i)
        if (relative[i] != (uint64_t{1} << i)) invalid = true;
      if (invalid) {
        op->emitError("layout changes a native vector packet");
        return;
      }
      SiteContract contract{bases.front(), packet, native.getAlignment(ptr), mask, {}, {}, {}};
      if (auto info = native.getAxisInfo(ptr)) {
        contract.contiguity.assign(info->getContiguity().begin(), info->getContiguity().end());
        contract.divisibility.assign(info->getDivisibility().begin(), info->getDivisibility().end());
        contract.constancy.assign(info->getConstancy().begin(), info->getConstancy().end());
        for (auto &value : contract.contiguity) value = std::min<int64_t>(value, packet);
        for (auto &value : contract.divisibility) value = std::min<int64_t>(value, packet * bytes);
      }
      contracts[op] = contract;
    });
    if (invalid) return signalPassFailure();
    if (inspect) {
      emitContracts(contracts);
      return;
    }
    OffsetBuilder offsets(&getContext(), bindings, mode == "repaired");
    SmallVector<scf::ForOp> loops;
    getOperation().walk<WalkOrder::PreOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
    for (auto loop : loops)
      if (failed(offsets.extendLoop(loop, byArgument))) return signalPassFailure();

    unsigned site = 0;
    SmallVector<LoadOp> loads;
    getOperation().walk([&](LoadOp load) { loads.push_back(load); });
    for (LoadOp load : loads) {
      auto found = contracts.find(load);
      if (found == contracts.end()) continue;
      const auto &contract = found->second;
      const auto &spec = byArgument.at(contract.argument);
      Value physical = offsets.physical(load.getPtr(), spec, modeBits(contract.packet));
      if (!physical) {
        load.emitError("unsupported structured pointer provenance (no pointer-integer fallback)");
        return signalPassFailure();
      }
      OpBuilder b(load);
      Location loc = load.getLoc();
      if (!contract.contiguity.empty()) {
        // Arithmetic reassociation and buffer conversion discard attributes on
        // addi/addptr. A pure tied-register identity carries these proven facts
        // through those passes; it emits no address arithmetic or memory access.
        auto target = getOperation()->getAttrOfType<StringAttr>("ttg.target");
        if (!target || (!target.getValue().starts_with("hip:") && !target.getValue().starts_with("cuda:"))) {
          load.emitError("packet proof carrier requires an AMD or NVIDIA target");
          return signalPassFailure();
        }
        auto carrier = ElementwiseInlineAsmOp::create(b, loc, TypeRange{physical.getType()},
            "", target.getValue().starts_with("hip:") ? "=v,0" : "=r,0", true, 1, ValueRange{physical});
        if (mode != "legacy") carrier->setAttr("laqs.packet_identity", b.getUnitAttr());
        physical = carrier->getResult(0);
        SmallVector<int64_t> divisibility(contract.divisibility);
        Type element = load.getType();
        if (auto tensor = dyn_cast<RankedTensorType>(element)) element = tensor.getElementType();
        unsigned bytes = (element.getIntOrFloatBitWidth() + 7) / 8;
        for (auto &value : divisibility) value = std::max<int64_t>(1, value / bytes);
        physical.getDefiningOp()->setAttr("tt.contiguity", hint(b, contract.contiguity));
        physical.getDefiningOp()->setAttr("tt.divisibility", hint(b, divisibility));
        physical.getDefiningOp()->setAttr("tt.constancy", hint(b, contract.constancy));
      }
      auto function = load->getParentOfType<FunctionOpInterface>();
      Value base = function.getArgument(contract.argument);
      Type ptrType = load.getPtr().getType(), scalar = ptrType;
      if (auto tensor = dyn_cast<RankedTensorType>(ptrType)) scalar = tensor.getElementType();
      if (base.getType() != scalar) base = BitcastOp::create(b, loc, scalar, base);
      if (isa<RankedTensorType>(ptrType)) base = SplatOp::create(b, loc, ptrType, base);
      auto pointer = AddPtrOp::create(b, loc, ptrType, base, physical);
      if (!contract.contiguity.empty()) {
        pointer->setAttr("tt.contiguity", hint(b, contract.contiguity));
        pointer->setAttr("tt.divisibility", hint(b, contract.divisibility));
        pointer->setAttr("tt.constancy", hint(b, contract.constancy));
      }
      load.getPtrMutable().assign(pointer.getResult());
      load->setAttr("laqs.packet_site", b.getI32IntegerAttr(site));
      ++site;
    }
    if (site == 0 && !specs.empty()) {
      getOperation().emitError("no supported packet load sites found");
      return signalPassFailure();
    }
    emitContracts(contracts);
    getOperation()->setAttr("laqs.physical_recurrences", IntegerAttr::get(IntegerType::get(&getContext(), 32), offsets.recurrenceCount()));
    ModuleAxisInfoAnalysis verified(getOperation());
    for (auto [op, contract] : contracts) {
      auto load = cast<LoadOp>(op);
      if (verified.getContiguity(load.getPtr()) < contract.packet) {
        load.emitError("packet contiguity was not established by the structured address proof");
        return signalPassFailure();
      }
    }
  }
private:
  static DenseIntElementsAttr hint(OpBuilder &b, ArrayRef<int64_t> values) {
    SmallVector<int32_t> narrow(values.begin(), values.end());
    return b.getI32TensorAttr(narrow);
  }
  void emitContracts(const llvm::DenseMap<Operation *, SiteContract> &contracts) {
    llvm::json::Array records;
    unsigned site = 0;
    getOperation().walk([&](LoadOp load) {
      auto found = contracts.find(load);
      if (found == contracts.end()) return;
      const auto &c = found->second;
      std::string ownership;
      llvm::raw_string_ostream stream(ownership);
      load.getType().print(stream);
      records.push_back(llvm::json::Object{{"site", site++}, {"argument", c.argument},
          {"vector_elements", c.packet}, {"mask_alignment", c.maskAlignment},
          {"native_alignment", c.alignment}, {"ownership", ownership}});
    });
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << llvm::json::Value(std::move(records));
    getOperation()->setAttr("laqs.packet_contract_json", StringAttr::get(&getContext(), text));
  }
  std::vector<LayoutSpec> specs;
  bool inspect = false;
  std::string mode = "repaired";
  ScalarBindings bindings;
};
} // namespace
} // namespace mlir::triton::laqs_packet

static void addPacketLayoutPass(mlir::PassManager *manager, const std::vector<std::string> &arguments) {
  std::vector<mlir::triton::laqs_packet::LayoutSpec> specs;
  bool inspect = false;
  std::string mode = "repaired";
  mlir::triton::laqs_packet::ScalarBindings bindings;
  for (const auto &argument : arguments) {
    if (argument == "inspect") inspect = true;
    else if (argument == "legacy" || argument == "transparent" || argument == "repaired") mode = argument;
    else if (llvm::StringRef(argument).starts_with("bind:")) {
      auto fields = mlir::triton::laqs_packet::splitFields(argument, ':');
      int64_t value;
      if (fields.size() != 3 || fields[2].getAsInteger(10, value) ||
          !bindings.emplace(fields[1].str(), value).second)
        llvm::report_fatal_error("invalid frozen scalar binding");
    }
    else specs.push_back(mlir::triton::laqs_packet::parseSpec(argument));
  }
  manager->addPass(std::make_unique<mlir::triton::laqs_packet::PacketLayoutPass>(std::move(specs), inspect, mode, bindings));
}
static void registerPacketLayoutPass() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<mlir::triton::laqs_packet::PacketLayoutPass>();
  });
}
TRITON_PLUGIN_API mlir::triton::plugin::PluginInfo *tritonGetPluginInfo() {
  static mlir::triton::plugin::PassInfo pass = {"laqs_packet_layout", "1.0.0", addPacketLayoutPass, registerPacketLayoutPass};
  static mlir::triton::plugin::PluginInfo info = {TRITON_PLUGIN_API_VERSION, "LAQSTritonPacketLayout", "1.0.0", &pass, 1, nullptr, 0, nullptr, 0, TRITON_VERSION};
  return &info;
}
