#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Tools/PluginUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace mlir::triton::laqs {
namespace {

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
  return name == "tt.splat" || name == "tt.broadcast" ||
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

Type i64Like(OpBuilder &builder, Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return RankedTensorType::get(tensor.getShape(), builder.getI64Type(),
                                 tensor.getEncoding());
  return builder.getI64Type();
}

Value constantLike(OpBuilder &builder, Location location, Type type,
                   uint64_t value) {
  auto scalar = builder.getIntegerAttr(builder.getI64Type(), value);
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

Value physicalOffset(OpBuilder &builder, Location location, Value elementOffset,
                     const LayoutSpec &spec) {
  Type type = elementOffset.getType();
  Value remaining = elementOffset;
  Value logical = constantLike(builder, location, type, 0);

  std::vector<unsigned> dimensions(spec.shape.size());
  std::iota(dimensions.begin(), dimensions.end(), 0);
  std::stable_sort(dimensions.begin(), dimensions.end(),
                   [&](unsigned a, unsigned b) {
                     return spec.strides[a] > spec.strides[b];
                   });
  std::vector<unsigned> shifts(spec.shape.size());
  unsigned shift = 0;
  for (unsigned dimension = 0; dimension < spec.shape.size(); ++dimension) {
    shifts[dimension] = shift;
    shift += modeBits(spec.shape[dimension]);
  }
  if (shift != spec.rows.size())
    llvm::report_fatal_error("LAQS row count does not match shape envelope");

  for (unsigned dimension : dimensions) {
    Value stride =
        constantLike(builder, location, type, spec.strides[dimension]);
    Value coordinate =
        arith::DivUIOp::create(builder, location, remaining, stride);
    remaining = arith::RemUIOp::create(builder, location, remaining, stride);
    if (shifts[dimension]) {
      Value amount = constantLike(builder, location, type, shifts[dimension]);
      coordinate = arith::ShLIOp::create(builder, location, coordinate, amount);
    }
    logical = arith::OrIOp::create(builder, location, logical, coordinate);
  }

  Value physical = constantLike(builder, location, type, 0);
  for (unsigned physicalBit = 0; physicalBit < spec.rows.size();
       ++physicalBit) {
    Value parity = arith::AndIOp::create(
        builder, location, logical,
        constantLike(builder, location, type, spec.rows[physicalBit]));
    for (unsigned fold : {32u, 16u, 8u, 4u, 2u, 1u}) {
      Value shifted =
          arith::ShRUIOp::create(builder, location, parity,
                                 constantLike(builder, location, type, fold));
      parity = arith::XOrIOp::create(builder, location, parity, shifted);
    }
    parity = arith::AndIOp::create(builder, location, parity,
                                   constantLike(builder, location, type, 1));
    if (physicalBit) {
      parity = arith::ShLIOp::create(
          builder, location, parity,
          constantLike(builder, location, type, physicalBit));
    }
    physical = arith::OrIOp::create(builder, location, physical, parity);
  }
  return physical;
}

class LayoutRewritePass
    : public PassWrapper<LayoutRewritePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutRewritePass)

  explicit LayoutRewritePass(std::vector<LayoutSpec> specs)
      : specs(std::move(specs)) {}
  LayoutRewritePass() = default;

  StringRef getArgument() const override { return "laqs-layout-rewrite"; }
  StringRef getDescription() const override {
    return "rewrite selected Triton pointer arguments through LAQS layouts";
  }

  void runOnOperation() override {
    std::map<unsigned, LayoutSpec> byArgument;
    for (const LayoutSpec &spec : specs) {
      if (!byArgument.emplace(spec.argument, spec).second)
        llvm::report_fatal_error("duplicate LAQS argument layout");
    }

    SmallVector<LoadOp> loads;
    getOperation().walk([&](LoadOp load) { loads.push_back(load); });
    unsigned rewritten = 0;
    for (LoadOp load : loads) {
      llvm::SetVector<unsigned> bases;
      collectBaseArguments(load.getPtr(), bases);
      if (bases.size() != 1)
        continue;
      auto found = byArgument.find(bases.front());
      if (found == byArgument.end())
        continue;

      auto function = load->getParentOfType<FunctionOpInterface>();
      if (!function || found->first >= function.getNumArguments())
        llvm::report_fatal_error("LAQS argument is absent from kernel");
      Value base = function.getArgument(found->first);
      auto pointer = dyn_cast<PointerType>(base.getType());
      if (!pointer)
        llvm::report_fatal_error("LAQS argument is not a scalar pointer");
      unsigned elementBits = pointer.getPointeeType().getIntOrFloatBitWidth();
      unsigned elementBytes = (elementBits + 7) / 8;
      if (elementBytes == 0 || (elementBytes & (elementBytes - 1)))
        llvm::report_fatal_error("LAQS supports power-of-two element bytes");

      OpBuilder builder(load);
      Location location = load.getLoc();
      Type pointerType = load.getPtr().getType();
      Type integerType = i64Like(builder, pointerType);
      Value baseLike = base;
      if (auto tensor = dyn_cast<RankedTensorType>(pointerType))
        baseLike = SplatOp::create(builder, location, tensor, base);
      Value address =
          PtrToIntOp::create(builder, location, integerType, load.getPtr());
      Value baseAddress =
          PtrToIntOp::create(builder, location, integerType, baseLike);
      Value byteOffset =
          arith::SubIOp::create(builder, location, address, baseAddress);
      Value elementOffset = byteOffset;
      if (elementBytes > 1) {
        unsigned byteShift = 0;
        while ((1u << byteShift) < elementBytes)
          ++byteShift;
        elementOffset = arith::ShRUIOp::create(
            builder, location, byteOffset,
            constantLike(builder, location, integerType, byteShift));
      }
      Value offset =
          physicalOffset(builder, location, elementOffset, found->second);
      Value newPointer =
          AddPtrOp::create(builder, location, pointerType, baseLike, offset);
      load.getPtrMutable().assign(newPointer);
      ++rewritten;
    }
    getOperation()->setAttr(
        "laqs.layout_rewrite_count",
        IntegerAttr::get(IntegerType::get(&getContext(), 64), rewritten));
  }

private:
  std::vector<LayoutSpec> specs;
};

std::unique_ptr<Pass> createLayoutRewritePass(std::vector<LayoutSpec> specs) {
  return std::make_unique<LayoutRewritePass>(std::move(specs));
}

} // namespace
} // namespace mlir::triton::laqs

static void addLayoutRewritePass(mlir::PassManager *manager,
                                 const std::vector<std::string> &arguments) {
  std::vector<mlir::triton::laqs::LayoutSpec> specs;
  specs.reserve(arguments.size());
  for (const std::string &argument : arguments)
    specs.push_back(mlir::triton::laqs::parseSpec(argument));
  manager->addPass(
      mlir::triton::laqs::createLayoutRewritePass(std::move(specs)));
}

static void registerLayoutRewritePass() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::triton::laqs::createLayoutRewritePass({});
  });
}

TRITON_PLUGIN_API mlir::triton::plugin::PluginInfo *tritonGetPluginInfo() {
  static mlir::triton::plugin::PassInfo pass = {"laqs_layout_rewrite", "1.0.0",
                                                addLayoutRewritePass,
                                                registerLayoutRewritePass};
  static mlir::triton::plugin::PluginInfo info = {
      TRITON_PLUGIN_API_VERSION,
      "LAQSTritonLayoutRewrite",
      "1.0.0",
      &pass,
      1,
      nullptr,
      0,
      nullptr,
      0,
      TRITON_VERSION,
  };
  return &info;
}
