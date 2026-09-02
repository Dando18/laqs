#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/PluginUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace mlir::triton::laqs {
namespace {

constexpr llvm::StringLiteral kManifestAttribute = "laqs.access_manifest";

std::string printType(Type type) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  type.print(stream);
  return stream.str();
}

std::string printAttribute(Attribute attribute) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  attribute.print(stream);
  return stream.str();
}

std::string printInteger(const APInt &value, bool isSigned) {
  SmallString<64> storage;
  value.toString(storage, 10, isSigned);
  return storage.str().str();
}

llvm::json::Array integerArray(ArrayRef<int64_t> values) {
  llvm::json::Array result;
  for (int64_t value : values)
    result.push_back(value);
  return result;
}

llvm::json::Array integerArray(ArrayRef<int32_t> values) {
  llvm::json::Array result;
  for (int32_t value : values)
    result.push_back(value);
  return result;
}

Type elementType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  if (auto pointer = dyn_cast<PointerType>(type))
    return pointer.getPointeeType();
  return type;
}

bool isPointerLike(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  return isa<PointerType>(type);
}

llvm::json::Array shapeOf(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return integerArray(tensor.getShape());
  if (auto descriptor = dyn_cast<TensorDescType>(type))
    return integerArray(descriptor.getShape());
  return {};
}

ArrayRef<int64_t> shapeRef(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.getShape();
  return {};
}

FileLineColLoc sourceLocation(Location location) {
  if (auto file = dyn_cast<FileLineColLoc>(location))
    return file;
  if (auto name = dyn_cast<NameLoc>(location))
    return sourceLocation(name.getChildLoc());
  if (auto opaque = dyn_cast<OpaqueLoc>(location))
    return sourceLocation(opaque.getFallbackLocation());
  if (auto fused = dyn_cast<FusedLoc>(location)) {
    if (!fused.getLocations().empty())
      return sourceLocation(fused.getLocations().front());
  }
  if (auto callSite = dyn_cast<CallSiteLoc>(location))
    return sourceLocation(callSite.getCallee());
  return FileLineColLoc::get(
      StringAttr::get(location.getContext(), "<unknown>"), 0, 0);
}

llvm::json::Object serializeLocation(Location location) {
  FileLineColLoc file = sourceLocation(location);
  return llvm::json::Object{{"file", file.getFilename().getValue().str()},
                            {"line", static_cast<int64_t>(file.getLine())},
                            {"column", static_cast<int64_t>(file.getColumn())}};
}

bool isDirectMemoryOp(Operation *operation) {
  return isa<LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp>(operation);
}

bool isDescriptorMemoryOp(Operation *operation) {
  return isa<DescriptorLoadOp, DescriptorStoreOp>(operation);
}

bool isUnsupportedDescriptorOp(Operation *operation) {
  return isa<DescriptorGatherOp, DescriptorScatterOp, DescriptorReduceOp>(
      operation);
}

bool isMemoryOp(Operation *operation) {
  return isDirectMemoryOp(operation) || isDescriptorMemoryOp(operation);
}

bool hasUnmodeledGlobalMemoryEffect(Operation *operation) {
  auto memory = dyn_cast<MemoryEffectOpInterface>(operation);
  if (!memory)
    return false;
  SmallVector<MemoryEffects::EffectInstance> effects;
  memory.getEffects(effects);
  return llvm::any_of(effects, [](const MemoryEffects::EffectInstance &effect) {
    return effect.getResource() == GlobalMemory::get();
  });
}

std::optional<StringRef> canonicalExpressionName(StringRef name) {
  return llvm::StringSwitch<std::optional<StringRef>>(name)
      .Case("tt.get_program_id", "program_id")
      .Case("tt.get_num_programs", "num_programs")
      .Case("tt.make_range", "make_range")
      .Case("tt.splat", "splat")
      .Case("tt.broadcast", "broadcast")
      .Case("tt.expand_dims", "expand_dims")
      .Case("tt.reshape", "reshape")
      .Case("tt.trans", "transpose")
      .Case("ttg.convert_layout", "convert_layout")
      .Case("arith.addi", "add")
      .Case("arith.subi", "sub")
      .Case("arith.muli", "mul")
      .Case("arith.shli", "shl")
      .Case("arith.shrui", "lshr")
      .Case("arith.shrsi", "ashr")
      .Case("arith.divsi", "sdiv")
      .Case("arith.divui", "udiv")
      .Case("arith.remsi", "srem")
      .Case("arith.remui", "urem")
      .Case("arith.andi", "and")
      .Case("arith.ori", "or")
      .Case("arith.xori", "xor")
      .Case("arith.cmpi", "cmp")
      .Case("arith.select", "select")
      .Case("arith.minsi", "smin")
      .Case("arith.minui", "umin")
      .Case("arith.maxsi", "smax")
      .Case("arith.maxui", "umax")
      .Case("arith.extsi", "sext")
      .Case("arith.extui", "zext")
      .Case("arith.trunci", "trunc")
      .Case("arith.bitcast", "bitcast")
      .Case("tt.bitcast", "bitcast")
      .Case("tt.ptr_to_int", "ptr_to_int")
      .Case("tt.int_to_ptr", "int_to_ptr")
      .Case("tt.addptr", "addptr")
      .Default(std::nullopt);
}

bool preservesPointerProvenance(StringRef name) {
  return name == "tt.splat" || name == "tt.broadcast" ||
         name == "tt.expand_dims" || name == "tt.reshape" ||
         name == "tt.trans" || name == "ttg.convert_layout" ||
         name == "builtin.unrealized_conversion_cast";
}

struct Diagnostic {
  std::string code;
  std::string message;
  Location location;
};

class ManifestBuilder {
public:
  explicit ManifestBuilder(ModuleOp module)
      : module(module), axisInfo(module), context(module.getContext()) {
    assignSiteIds();
  }

  llvm::json::Object build() {
    llvm::json::Array arguments;
    llvm::json::Array body;
    unsigned functionCount = 0;
    for (Operation &operation : module.getBody()->getOperations()) {
      auto function = dyn_cast<FunctionOpInterface>(&operation);
      if (!function)
        continue;
      if (functionCount++ != 0) {
        addDiagnostic("unsupported.multiple_functions",
                      "the post-inline manifest requires one kernel function",
                      operation.getLoc());
        continue;
      }
      serializeFunctionArguments(function, arguments);
      body = serializeRegion(function.getFunctionBody());
    }
    if (functionCount == 0)
      addDiagnostic("unsupported.missing_function",
                    "the module contains no kernel function", module.getLoc());

    llvm::json::Array expressionArray;
    for (llvm::json::Object &expression : expressions)
      expressionArray.push_back(std::move(expression));
    llvm::json::Array layoutArray;
    for (llvm::json::Object &layout : layouts)
      layoutArray.push_back(std::move(layout));
    llvm::json::Array diagnosticArray;
    for (const Diagnostic &diagnostic : diagnostics) {
      diagnosticArray.push_back(llvm::json::Object{
          {"category", diagnostic.code},
          {"message", diagnostic.message},
          {"source", serializeLocation(diagnostic.location)}});
    }

    llvm::json::Object root;
    root["schema"] = "laqs.triton.access_manifest";
    root["version"] = 1;
    root["status"] = diagnostics.empty() ? "supported" : "unsupported";
    root["args"] = std::move(arguments);
    root["layouts"] = std::move(layoutArray);
    root["expressions"] = std::move(expressionArray);
    root["body"] = std::move(body);
    root["diagnostics"] = std::move(diagnosticArray);
    return root;
  }

private:
  struct ArgumentBinding {
    std::string name;
    llvm::json::Array path;
  };

  ArgumentBinding argumentBinding(BlockArgument argument) {
    auto nameLocation = dyn_cast<NameLoc>(argument.getLoc());
    if (!nameLocation) {
      addDiagnostic("unsupported.argument_binding",
                    "a specialized function argument has no source NameLoc",
                    argument.getLoc());
      return {(Twine("arg") + Twine(argument.getArgNumber())).str(), {}};
    }
    SmallVector<StringRef> components;
    nameLocation.getName().getValue().split(components, '.');
    ArgumentBinding result{components.front().str(), {}};
    for (StringRef component : llvm::drop_begin(components)) {
      int64_t index;
      if (!component.getAsInteger(10, index)) {
        result.path.push_back(index);
      } else if (component == "stride") {
        result.path.push_back("strides");
      } else {
        result.path.push_back(component.str());
      }
    }
    return result;
  }

  ArgumentBinding argumentBinding(unsigned index) {
    auto found = functionArguments.find(index);
    if (found == functionArguments.end()) {
      addDiagnostic("unsupported.argument_binding",
                    Twine("unknown specialized argument index ") + Twine(index),
                    module.getLoc());
      return {(Twine("arg") + Twine(index)).str(), {}};
    }
    return argumentBinding(found->second);
  }

  void assignSiteIds() {
    int64_t index = 0;
    module.walk([&](Operation *operation) {
      if (!isMemoryOp(operation) && !isUnsupportedDescriptorOp(operation))
        return;
      std::string id;
      llvm::raw_string_ostream stream(id);
      stream << "memory." << index++;
      siteIds[operation] = stream.str();
    });
  }

  void addDiagnostic(StringRef code, const Twine &message, Location location) {
    std::string text = message.str();
    std::pair<std::string, std::string> key(code.str(), text);
    if (diagnosticKeys.insert(key).second)
      diagnostics.push_back({code.str(), std::move(text), location});
  }

  void serializeFunctionArguments(FunctionOpInterface function,
                                  llvm::json::Array &arguments) {
    for (auto [index, argument] : llvm::enumerate(function.getArguments())) {
      functionArguments[index] = argument;
      ArgumentBinding binding = argumentBinding(argument);
      llvm::json::Object object;
      object["index"] = static_cast<int64_t>(index);
      object["name"] = binding.name;
      object["path"] = std::move(binding.path);
      object["function"] = function.getName().str();
      object["expression"] = expression(argument);
      object["type"] = printType(argument.getType());
      object["shape"] = shapeOf(argument.getType());
      Type scalar = elementType(argument.getType());
      object["element_type"] = printType(scalar);
      Type pointerCandidate = argument.getType();
      if (auto tensor = dyn_cast<RankedTensorType>(pointerCandidate))
        pointerCandidate = tensor.getElementType();
      if (isa<PointerType>(pointerCandidate))
        object["kind"] = "pointer";
      else if (isa<TensorDescType>(argument.getType()))
        object["kind"] = "tensor_descriptor";
      else
        object["kind"] = "scalar";
      arguments.push_back(std::move(object));
    }
  }

  llvm::json::Array serializeRegion(Region &region) {
    llvm::json::Array result;
    for (Block &block : region) {
      for (Operation &operation : block) {
        if (isa<scf::YieldOp, triton::ReturnOp>(operation))
          continue;
        if (auto loop = dyn_cast<scf::ForOp>(operation)) {
          result.push_back(serializeFor(loop));
          continue;
        }
        if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
          result.push_back(serializeIf(conditional));
          continue;
        }
        if (isa<scf::WhileOp>(operation)) {
          addDiagnostic("unsupported.data_dependent_while",
                        "scf.while does not have a statically bounded exact "
                        "trace contract",
                        operation.getLoc());
          continue;
        }
        if (isMemoryOp(&operation)) {
          if (auto memory = serializeMemory(&operation))
            result.push_back(std::move(*memory));
          continue;
        }
        if (isUnsupportedDescriptorOp(&operation)) {
          addDiagnostic("unsupported.descriptor_operation",
                        Twine(operation.getName().getStringRef()) +
                            " is outside the exact descriptor subset",
                        operation.getLoc());
          continue;
        }
        StringRef name = operation.getName().getStringRef();
        if (name.contains("barrier") || name.contains("fence")) {
          result.push_back(llvm::json::Object{
              {"kind", "barrier"},
              {"operation", name.str()},
              {"source", serializeLocation(operation.getLoc())}});
          continue;
        }
        if (auto inlineAssembly = dyn_cast<ElementwiseInlineAsmOp>(operation);
            inlineAssembly && !inlineAssembly.getPure()) {
          addDiagnostic("unsupported.opaque_memory_operation",
                        "impure inline assembly has opaque memory effects",
                        operation.getLoc());
          continue;
        }
        if (auto external = dyn_cast<ExternElementwiseOp>(operation);
            external && !external.getPure()) {
          addDiagnostic("unsupported.opaque_memory_operation",
                        "impure external elementwise code has opaque memory "
                        "effects",
                        operation.getLoc());
          continue;
        }
        if (hasUnmodeledGlobalMemoryEffect(&operation)) {
          addDiagnostic("unsupported.custom_global_memory_operation",
                        Twine("unmodeled global-memory effect in ") + name,
                        operation.getLoc());
        }
      }
    }
    return result;
  }

  llvm::json::Object serializeFor(scf::ForOp loop) {
    std::string id = (Twine("loop.") + Twine(nextLoop++)).str();
    loopIds[loop.getOperation()] = id;
    activeLoops.push_back(id);

    llvm::json::Object object;
    object["kind"] = "for";
    object["iv"] = id + ".iv";
    object["lower"] = expression(loop.getLowerBound());
    object["upper"] = expression(loop.getUpperBound());
    object["step"] = expression(loop.getStep());
    object["source"] = serializeLocation(loop.getLoc());
    object["lexical_order"] = lexicalOrder++;
    object["body"] = serializeRegion(loop.getRegion());

    llvm::json::Array carried;
    Operation *terminator = loop.getBody()->getTerminator();
    for (auto [slot, initial] : llvm::enumerate(loop.getInitArgs())) {
      Value regionArgument = loop.getRegionIterArg(slot);
      if (!expressionIds.contains(regionArgument) &&
          !pointerOffsetIds.contains(regionArgument))
        continue;
      llvm::json::Object item;
      item["name"] = id + ".iter" + Twine(slot).str();
      bool pointer = isPointerLike(regionArgument.getType());
      item["init"] =
          pointer ? pointerOffset(initial, loop.getLoc()) : expression(initial);
      item["yield"] =
          pointer ? pointerOffset(terminator->getOperand(slot), loop.getLoc())
                  : expression(terminator->getOperand(slot));
      carried.push_back(std::move(item));
    }
    object["iter_args"] = std::move(carried);
    activeLoops.pop_back();
    return object;
  }

  llvm::json::Object serializeIf(scf::IfOp conditional) {
    int64_t predicate = expression(conditional.getCondition());
    llvm::json::Object object;
    object["kind"] = "if";
    object["condition"] = predicate;
    object["source"] = serializeLocation(conditional.getLoc());
    object["lexical_order"] = lexicalOrder++;
    controlPredicates.push_back({predicate, true});
    object["then"] = serializeRegion(conditional.getThenRegion());
    controlPredicates.pop_back();
    if (!conditional.getElseRegion().empty()) {
      controlPredicates.push_back(
          {negate(predicate, conditional.getLoc()), true});
      object["else"] = serializeRegion(conditional.getElseRegion());
      controlPredicates.pop_back();
    } else {
      object["else"] = llvm::json::Array();
    }
    return object;
  }

  int64_t syntheticExpression(StringRef op, StringRef type,
                              ArrayRef<int64_t> operands,
                              llvm::json::Object attributes, Location location,
                              ArrayRef<int64_t> shape = {}) {
    int64_t id = expressions.size();
    llvm::json::Object object;
    object["id"] = id;
    object["op"] = op.str();
    object["type"] = type.str();
    object["operands"] = integerArray(operands);
    attributes["integer_width"] = 64;
    object["attributes"] = std::move(attributes);
    object["source"] = serializeLocation(location);
    object["shape"] = integerArray(shape);
    expressions.push_back(std::move(object));
    unsigned depth = 0;
    for (int64_t operand : operands)
      depth = std::max(depth, expressionDataDepth.lookup(operand));
    expressionDataDepth[id] = depth;
    expressionIntegerWidths[id] = 64;
    return id;
  }

  int64_t widenPointerOffset(int64_t expressionId, Type carrierType,
                             Location location) {
    unsigned width = expressionIntegerWidths.lookup(expressionId);
    if (width == 64)
      return expressionId;
    if (width == 0 || width > 64) {
      addDiagnostic("unsupported.pointer_offset_width",
                    "pointer offset is not an integer of at most 64 bits",
                    location);
      return expressionId;
    }
    return syntheticExpression("sext", "i64", {expressionId}, {}, location,
                               shapeRef(carrierType));
  }

  int64_t zero(Location location) {
    int64_t result = syntheticExpression(
        "constant", "i64", {}, llvm::json::Object{{"value", "0"}}, location);
    zeroExpressions.insert(result);
    return result;
  }

  int64_t negate(int64_t predicate, Location location) {
    int64_t falseValue = zero(location);
    int64_t operands[] = {predicate, falseValue};
    return syntheticExpression("cmp", "i1", operands,
                               llvm::json::Object{{"predicate", "eq"}},
                               location);
  }

  int64_t pointerOffset(Value value, Location useLocation) {
    auto found = pointerOffsetIds.find(value);
    if (found != pointerOffsetIds.end())
      return found->second;

    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Block *block = argument.getOwner();
      Operation *parent = block->getParentOp();
      if (isa<FunctionOpInterface>(parent)) {
        if (isa<RankedTensorType>(argument.getType()))
          addDiagnostic("unsupported.tensor_pointer_argument",
                        "a launch pointer argument must be scalar",
                        useLocation);
        int64_t result = zero(useLocation);
        pointerOffsetIds[value] = result;
        return result;
      }
      if (auto loop = dyn_cast<scf::ForOp>(parent)) {
        if (argument == loop.getInductionVar()) {
          addDiagnostic("unsupported.pointer_provenance",
                        "loop induction variable used as a pointer",
                        useLocation);
          int64_t result = zero(useLocation);
          pointerOffsetIds[value] = result;
          return result;
        }
        auto loopId = loopIds.find(parent);
        if (loopId == loopIds.end()) {
          std::string id = (Twine("loop.") + Twine(nextLoop++)).str();
          loopIds[parent] = id;
          loopId = loopIds.find(parent);
        }
        std::string name =
            loopId->second + ".iter" + Twine(argument.getArgNumber() - 1).str();
        int64_t result = syntheticExpression(
            "loop_carried",
            isa<RankedTensorType>(argument.getType()) ? "tensor" : "i64", {},
            llvm::json::Object{{"name", name}}, useLocation,
            dyn_cast<RankedTensorType>(argument.getType())
                ? dyn_cast<RankedTensorType>(argument.getType()).getShape()
                : ArrayRef<int64_t>());
        pointerOffsetIds[value] = result;
        return result;
      }
      addDiagnostic("unsupported.pointer_provenance",
                    "pointer flows through an unsupported block argument",
                    useLocation);
      int64_t result = zero(useLocation);
      pointerOffsetIds[value] = result;
      return result;
    }

    Operation *definition = value.getDefiningOp();
    StringRef name = definition->getName().getStringRef();
    int64_t result;
    if (name == "tt.addptr") {
      int64_t base = pointerOffset(definition->getOperand(0), useLocation);
      int64_t delta = expression(definition->getOperand(1));
      if (zeroExpressions.contains(base)) {
        result = delta;
      } else {
        base = widenPointerOffset(base, definition->getOperand(0).getType(),
                                  definition->getLoc());
        delta = widenPointerOffset(delta, definition->getOperand(1).getType(),
                                   definition->getLoc());
        int64_t operands[] = {base, delta};
        result = syntheticExpression(
            "add", isa<RankedTensorType>(value.getType()) ? "tensor" : "i64",
            operands, {}, definition->getLoc(), shapeRef(value.getType()));
      }
    } else if (preservesPointerProvenance(name)) {
      auto canonical = canonicalExpressionName(name);
      if (!canonical)
        canonical = StringRef("convert_layout");
      int64_t operand = pointerOffset(definition->getOperand(0), useLocation);
      if (zeroExpressions.contains(operand)) {
        result = operand;
      } else if (name == "ttg.convert_layout" ||
                 name == "builtin.unrealized_conversion_cast") {
        result = operand;
      } else {
        operand = widenPointerOffset(
            operand, definition->getOperand(0).getType(), definition->getLoc());
        result = syntheticExpression(
            *canonical,
            isa<RankedTensorType>(value.getType()) ? "tensor" : "i64",
            ArrayRef<int64_t>(&operand, 1),
            serializeExpressionAttributes(definition), definition->getLoc(),
            shapeRef(value.getType()));
      }
    } else if (name == "arith.select") {
      int64_t trueOffset =
          pointerOffset(definition->getOperand(1), useLocation);
      int64_t falseOffset =
          pointerOffset(definition->getOperand(2), useLocation);
      trueOffset =
          widenPointerOffset(trueOffset, definition->getOperand(1).getType(),
                             definition->getLoc());
      falseOffset =
          widenPointerOffset(falseOffset, definition->getOperand(2).getType(),
                             definition->getLoc());
      int64_t operands[] = {expression(definition->getOperand(0)), trueOffset,
                            falseOffset};
      result = syntheticExpression(
          "select", isa<RankedTensorType>(value.getType()) ? "tensor" : "i64",
          operands, {}, definition->getLoc(), shapeRef(value.getType()));
    } else {
      addDiagnostic("unsupported.pointer_offset",
                    Twine("cannot express a base-relative offset through ") +
                        name,
                    useLocation);
      result = zero(useLocation);
    }
    pointerOffsetIds[value] = result;
    return result;
  }

  int64_t descriptorOffset(Value descriptor, ValueRange indices, Type blockType,
                           Location location) {
    SmallVector<Value> strides;
    Value descriptorBase;
    std::optional<unsigned> descriptorArgument;
    if (auto make = descriptor.getDefiningOp<MakeTensorDescOp>()) {
      descriptorBase = make.getBase();
      llvm::append_range(strides, make.getStrides());
    } else if (auto argument = dyn_cast<BlockArgument>(descriptor);
               argument &&
               isa<FunctionOpInterface>(argument.getOwner()->getParentOp())) {
      descriptorArgument = argument.getArgNumber();
    } else {
      addDiagnostic("unsupported.descriptor_provenance",
                    "descriptor must be a launch argument or a direct "
                    "tt.make_tensor_descriptor result",
                    location);
      return zero(location);
    }

    auto tensor = dyn_cast<RankedTensorType>(blockType);
    if (!tensor || static_cast<size_t>(tensor.getRank()) != indices.size()) {
      addDiagnostic("unsupported.descriptor_rank",
                    "descriptor indices and block tensor rank disagree",
                    location);
      return zero(location);
    }
    if (!descriptorArgument && strides.size() != indices.size()) {
      addDiagnostic("unsupported.descriptor_rank",
                    "descriptor strides and indices disagree", location);
      return zero(location);
    }

    int64_t total = descriptorBase ? pointerOffset(descriptorBase, location)
                                   : zero(location);
    for (auto [dimension, index] : llvm::enumerate(indices)) {
      int64_t stride;
      if (descriptorArgument) {
        ArgumentBinding binding = argumentBinding(*descriptorArgument);
        binding.path.push_back("strides");
        binding.path.push_back(static_cast<int64_t>(dimension));
        stride = syntheticExpression(
            "arg", "i64", {},
            llvm::json::Object{
                {"arg", binding.name},
                {"index", static_cast<int64_t>(*descriptorArgument)},
                {"name", binding.name},
                {"path", std::move(binding.path)}},
            location);
      } else {
        stride = expression(strides[dimension]);
      }

      int64_t coordinate =
          widenPointerOffset(expression(index), index.getType(), location);
      int64_t extent = tensor.getShape()[dimension];
      int64_t range =
          syntheticExpression("make_range", "tensor", {},
                              llvm::json::Object{{"start", 0}, {"end", extent}},
                              location, ArrayRef<int64_t>(&extent, 1));
      SmallVector<int64_t> reshaped(tensor.getRank(), 1);
      reshaped[dimension] = extent;
      range = syntheticExpression("reshape", "tensor", {range}, {}, location,
                                  reshaped);
      range = syntheticExpression("broadcast", "tensor", {range}, {}, location,
                                  tensor.getShape());
      int64_t coordinateOperands[] = {coordinate, range};
      coordinate = syntheticExpression("add", "tensor", coordinateOperands, {},
                                       location, tensor.getShape());
      int64_t productOperands[] = {coordinate, stride};
      int64_t product = syntheticExpression("mul", "tensor", productOperands,
                                            {}, location, tensor.getShape());
      int64_t sumOperands[] = {total, product};
      total = syntheticExpression("add", "tensor", sumOperands, {}, location,
                                  tensor.getShape());
    }
    return total;
  }

  std::optional<llvm::json::Object> serializeMemory(Operation *operation) {
    Value address;
    Value mask;
    Value descriptor;
    ValueRange indices;
    Type distributedType;
    StringRef kind;

    if (auto load = dyn_cast<LoadOp>(operation)) {
      address = load.getPtr();
      mask = load.getMask();
      distributedType = address.getType();
      kind = "load";
    } else if (auto store = dyn_cast<StoreOp>(operation)) {
      address = store.getPtr();
      mask = store.getMask();
      distributedType = address.getType();
      kind = "store";
    } else if (auto atomic = dyn_cast<AtomicRMWOp>(operation)) {
      address = atomic.getPtr();
      mask = atomic.getMask();
      distributedType = address.getType();
      kind = "atomic";
    } else if (auto atomic = dyn_cast<AtomicCASOp>(operation)) {
      address = atomic.getPtr();
      distributedType = address.getType();
      kind = "atomic";
    } else if (auto load = dyn_cast<DescriptorLoadOp>(operation)) {
      descriptor = load.getDesc();
      indices = load.getIndices();
      distributedType = load.getResult().getType();
      kind = "load";
    } else {
      auto store = cast<DescriptorStoreOp>(operation);
      descriptor = store.getDesc();
      indices = store.getIndices();
      distributedType = store.getSrc().getType();
      kind = "store";
    }

    auto tensorType = dyn_cast<RankedTensorType>(distributedType);
    if (tensorType && !tensorType.getEncoding()) {
      addDiagnostic("unsupported.missing_memory_layout",
                    Twine(siteIds.lookup(operation)) + " has no encoding",
                    operation->getLoc());
      return std::nullopt;
    }

    Value provenanceValue = descriptor ? descriptor : address;
    llvm::SetVector<unsigned> bases;
    collectBaseArguments(provenanceValue, bases, operation->getLoc());
    if (bases.empty()) {
      addDiagnostic("unsupported.pointer_provenance",
                    Twine(siteIds.lookup(operation)) +
                        " does not resolve to a launch argument",
                    operation->getLoc());
    } else if (bases.size() > 1) {
      addDiagnostic("unsupported.ambiguous_pointer_provenance",
                    Twine(siteIds.lookup(operation)) +
                        " may address multiple base allocations",
                    operation->getLoc());
    }

    llvm::json::Object base;
    if (bases.size() == 1) {
      unsigned index = bases.front();
      ArgumentBinding binding = argumentBinding(index);
      base["arg_index"] = static_cast<int64_t>(index);
      base["arg"] = binding.name;
      base["path"] = std::move(binding.path);
    }

    llvm::json::Object object;
    object["kind"] = "memory";
    object["site_id"] = siteIds.lookup(operation);
    object["parent_operation_id"] = siteIds.lookup(operation);
    object["op"] = kind;
    object["source"] = serializeLocation(operation->getLoc());
    object["base"] = std::move(base);
    object["offset"] =
        address
            ? llvm::json::Value(pointerOffset(address, operation->getLoc()))
            : llvm::json::Value(descriptorOffset(
                  descriptor, indices, distributedType, operation->getLoc()));
    llvm::json::Array indexExpressions;
    for (Value index : indices)
      indexExpressions.push_back(expression(index));
    object["indices"] = std::move(indexExpressions);
    object["mask"] =
        mask ? llvm::json::Value(expression(mask)) : llvm::json::Value(nullptr);
    object["shape"] = tensorType ? llvm::json::Value(shapeOf(distributedType))
                                 : llvm::json::Value(llvm::json::Array{1});
    Type scalar = elementType(address ? address.getType() : distributedType);
    object["element_type"] = printType(scalar);
    unsigned bits = isa<IntegerType, FloatType>(scalar)
                        ? scalar.getIntOrFloatBitWidth()
                        : 0;
    object["element_bytes"] = static_cast<int64_t>((bits + 7) / 8);
    object["layout"] =
        tensorType ? layoutForType(distributedType) : scalarLayout();
    object["lexical_order"] = lexicalOrder++;

    llvm::json::Array predicates;
    for (auto [predicate, polarity] : controlPredicates) {
      assert(polarity && "false predicates are normalized by negate()");
      predicates.push_back(predicate);
    }
    object["control_predicates"] = std::move(predicates);
    llvm::json::Array loops;
    for (const std::string &loop : activeLoops)
      loops.push_back(loop);
    object["loops"] = std::move(loops);

    llvm::json::Object attributes;
    if (Attribute cache = operation->getAttr("cache"))
      attributes["cache"] = printAttribute(cache);
    if (Attribute eviction = operation->getAttr("evict"))
      attributes["eviction"] = printAttribute(eviction);
    if (Attribute semantic = operation->getAttr("sem"))
      attributes["memory_semantic"] = printAttribute(semantic);
    if (Attribute scope = operation->getAttr("scope"))
      attributes["memory_scope"] = printAttribute(scope);
    if (Attribute atomic = operation->getAttr("atomic_rmw_op"))
      attributes["atomic_operation"] = printAttribute(atomic);
    if (auto cache = attributes.getString("cache"))
      object["cache"] = cache->str();
    if (auto eviction = attributes.getString("eviction"))
      object["eviction"] = eviction->str();
    object["attributes"] = std::move(attributes);
    object["issue"] = serializeIssue(operation, address, distributedType);
    return object;
  }

  llvm::json::Object serializeIssue(Operation *operation, Value address,
                                    Type distributedType) {
    llvm::json::Object issue;
    issue["partition"] = "conservative_register_slices";
    issue["register_slice_elements"] = 1;
    issue["register_slice_size"] = 1;
    issue["common_parent_operation"] = siteIds.lookup(operation);

    if (auto tensor = dyn_cast<RankedTensorType>(distributedType)) {
      LinearLayout layout = gpu::toLinearLayout(tensor);
      StringAttr reg = StringAttr::get(context, "register");
      if (layout.hasInDim(reg))
        issue["register_elements_per_owner"] = layout.getInDimSize(reg);
      issue["layout_consecutive_elements"] = layout.getNumConsecutiveInOut();
    }

    if (address && isa<RankedTensorType>(address.getType())) {
      if (AxisInfo *info = axisInfo.getAxisInfo(address)) {
        issue["axis_contiguity"] = integerArray(info->getContiguity());
        issue["axis_divisibility"] = integerArray(info->getDivisibility());
        issue["axis_constancy"] = integerArray(info->getConstancy());
      }
      issue["alignment_elements"] =
          static_cast<int64_t>(axisInfo.getAlignment(address));
      if (auto predicated = dyn_cast<PredicatedOpInterface>(operation)) {
        Value operationMask = predicated.getPredicateOperand();
        if (operationMask)
          issue["mask_alignment_elements"] =
              static_cast<int64_t>(axisInfo.getMaskAlignment(operationMask));
      }
    }
    return issue;
  }

  int64_t expression(Value value) {
    auto found = expressionIds.find(value);
    if (found != expressionIds.end())
      return found->second;

    int64_t id = expressions.size();
    expressionIds[value] = id;
    expressions.emplace_back();
    llvm::json::Object object;
    object["id"] = id;
    object["type"] = printType(value.getType());
    object["shape"] = shapeOf(value.getType());
    object["layout"] = layoutForType(value.getType());
    if (auto integer = dyn_cast<IntegerType>(elementType(value.getType())))
      expressionIntegerWidths[id] = integer.getWidth();

    if (auto argument = dyn_cast<BlockArgument>(value)) {
      serializeBlockArgument(argument, object);
      expressionDataDepth[id] = 0;
      expressions[id] = std::move(object);
      return id;
    }

    Operation *operation = value.getDefiningOp();
    object["source"] = serializeLocation(operation->getLoc());
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      object["op"] = "constant";
      object["attributes"] = serializeConstant(constant.getValue());
      expressionDataDepth[id] = 0;
    } else if (isMemoryOp(operation)) {
      Type scalar = elementType(value.getType());
      auto load = dyn_cast<LoadOp>(operation);
      if (!load || !isa<IntegerType>(scalar)) {
        object["op"] = "unsupported";
        object["attributes"] =
            llvm::json::Object{{"site_id", siteIds.lookup(operation)}};
        addDiagnostic("unsupported.data_dependent_expression",
                      Twine(siteIds.lookup(operation)) +
                          " feeds an address, mask, or scalar branch through " +
                          printType(scalar),
                      operation->getLoc());
        expressionDataDepth[id] = 0;
      } else {
        llvm::SetVector<unsigned> bases;
        collectBaseArguments(load.getPtr(), bases, operation->getLoc());
        if (load.getMask())
          addDiagnostic("unsupported.masked_data_dependent_index",
                        Twine(siteIds.lookup(operation)) +
                            " is a masked integer indirection load",
                        operation->getLoc());
        if (bases.size() != 1) {
          object["op"] = "unsupported";
          addDiagnostic("unsupported.data_dependent_index",
                        "integer indirection has unresolved provenance",
                        operation->getLoc());
          expressionDataDepth[id] = 0;
        } else {
          ArgumentBinding binding = argumentBinding(bases.front());
          if (!binding.path.empty())
            addDiagnostic("unsupported.data_dependent_index",
                          "integer indirection through a flattened argument "
                          "path is unsupported",
                          operation->getLoc());
          int64_t offset = pointerOffset(load.getPtr(), operation->getLoc());
          unsigned operandDepth = expressionDataDepth.lookup(offset);
          if (operandDepth != 0)
            addDiagnostic("unsupported.pointer_chasing",
                          "more than one level of loaded integer indexing",
                          operation->getLoc());
          object["op"] = "gather";
          object["operands"] = integerArray(ArrayRef<int64_t>(&offset, 1));
          object["attributes"] = llvm::json::Object{
              {"arg", binding.name},
              {"path", std::move(binding.path)},
              {"site_id", siteIds.lookup(operation)},
              {"integer_width",
               static_cast<int64_t>(cast<IntegerType>(scalar).getWidth())}};
          expressionDataDepth[id] = operandDepth + 1;
        }
      }
    } else if (auto canonical = canonicalExpressionName(
                   operation->getName().getStringRef())) {
      object["op"] = *canonical;
      llvm::json::Array operands;
      for (Value operand : operation->getOperands())
        operands.push_back(expression(operand));
      object["operands"] = std::move(operands);
      object["attributes"] = serializeExpressionAttributes(operation);
      unsigned depth = 0;
      for (Value operand : operation->getOperands())
        depth = std::max(
            depth, expressionDataDepth.lookup(expressionIds.lookup(operand)));
      expressionDataDepth[id] = depth;
    } else {
      object["op"] = "unsupported";
      object["attributes"] = llvm::json::Object{
          {"operation", operation->getName().getStringRef()}};
      addDiagnostic("unsupported.expression_operation",
                    Twine("expression depends on ") +
                        operation->getName().getStringRef(),
                    operation->getLoc());
      expressionDataDepth[id] = 0;
    }
    expressions[id] = std::move(object);
    return id;
  }

  void serializeBlockArgument(BlockArgument argument,
                              llvm::json::Object &object) {
    Block *block = argument.getOwner();
    Operation *parent = block->getParentOp();
    if (auto function = dyn_cast<FunctionOpInterface>(parent)) {
      ArgumentBinding binding = argumentBinding(argument);
      object["op"] = "arg";
      object["attributes"] = llvm::json::Object{
          {"function", function.getName().str()},
          {"arg", binding.name},
          {"index", static_cast<int64_t>(argument.getArgNumber())},
          {"name", binding.name},
          {"path", std::move(binding.path)}};
      return;
    }
    if (auto loop = dyn_cast<scf::ForOp>(parent)) {
      auto loopId = loopIds.find(parent);
      if (loopId == loopIds.end()) {
        std::string id = (Twine("loop.") + Twine(nextLoop++)).str();
        loopIds[parent] = id;
        loopId = loopIds.find(parent);
      }
      if (argument == loop.getInductionVar()) {
        object["op"] = "loop_iv";
        object["attributes"] =
            llvm::json::Object{{"name", loopId->second + ".iv"}};
      } else {
        object["op"] = "loop_carried";
        object["attributes"] = llvm::json::Object{
            {"name", loopId->second + ".iter" +
                         Twine(argument.getArgNumber() - 1).str()}};
      }
      return;
    }
    object["op"] = "unsupported";
    object["attributes"] = llvm::json::Object{
        {"block_argument", static_cast<int64_t>(argument.getArgNumber())}};
    addDiagnostic("unsupported.block_argument",
                  "non-function, non-scf.for block argument in expression",
                  parent->getLoc());
  }

  llvm::json::Object serializeConstant(Attribute value) {
    llvm::json::Object object;
    object["spelling"] = printAttribute(value);
    if (auto integer = dyn_cast<IntegerAttr>(value)) {
      object["kind"] = "integer";
      object["value"] = printInteger(integer.getValue(), true);
      object["unsigned_value"] = printInteger(integer.getValue(), false);
      object["bit_width"] =
          static_cast<int64_t>(integer.getValue().getBitWidth());
    } else if (auto dense = dyn_cast<DenseIntElementsAttr>(value)) {
      object["kind"] = "integer_tensor";
      llvm::json::Array values;
      for (const APInt &item : dense.getValues<APInt>())
        values.push_back(printInteger(item, true));
      object["values"] = std::move(values);
      object["splat"] = dense.isSplat();
      object["bit_width"] =
          static_cast<int64_t>(dense.getElementType().getIntOrFloatBitWidth());
    } else {
      object["kind"] = "unsupported";
      addDiagnostic("unsupported.constant",
                    Twine("address expression uses constant ") +
                        printAttribute(value),
                    module.getLoc());
    }
    return object;
  }

  llvm::json::Object serializeExpressionAttributes(Operation *operation) {
    llvm::json::Object attributes;
    StringRef name = operation->getName().getStringRef();
    if (auto programId = dyn_cast<GetProgramIdOp>(operation)) {
      attributes["axis"] = programId.getAxisAsInt();
    } else if (auto numPrograms = dyn_cast<GetNumProgramsOp>(operation)) {
      attributes["axis"] = numPrograms.getAxisAsInt();
    } else if (name == "tt.make_range") {
      attributes["start"] =
          operation->getAttrOfType<IntegerAttr>("start").getInt();
      attributes["end"] = operation->getAttrOfType<IntegerAttr>("end").getInt();
    } else if (name == "tt.expand_dims") {
      attributes["axis"] =
          operation->getAttrOfType<IntegerAttr>("axis").getInt();
    } else if (name == "tt.trans") {
      if (auto order = operation->getAttrOfType<DenseI32ArrayAttr>("order"))
        attributes["order"] = integerArray(order.asArrayRef());
    } else if (auto compare = dyn_cast<arith::CmpIOp>(operation)) {
      attributes["predicate"] =
          arith::stringifyCmpIPredicate(compare.getPredicate()).str();
    }
    Type scalar = elementType(operation->getResult(0).getType());
    if (auto integer = dyn_cast<IntegerType>(scalar))
      attributes["integer_width"] = static_cast<int64_t>(integer.getWidth());
    return attributes;
  }

  llvm::json::Value layoutForType(Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.getEncoding())
      return nullptr;

    Type layoutType = tensor;
    auto found = layoutIds.find(layoutType);
    if (found != layoutIds.end())
      return found->second;

    std::string id = (Twine("layout.") + Twine(layouts.size())).str();
    layoutIds[layoutType] = id;
    LinearLayout layout = gpu::toLinearLayout(tensor);
    llvm::json::Object object;
    object["id"] = id;

    llvm::json::Array bases;
    for (const auto &[input, inputBases] : layout.getBases()) {
      llvm::json::Object inputObject;
      inputObject["input"] = input.getValue();
      llvm::json::Array basisArray;
      for (const std::vector<int32_t> &basis : inputBases)
        basisArray.push_back(integerArray(basis));
      inputObject["basis"] = std::move(basisArray);
      bases.push_back(std::move(inputObject));
    }
    object["bases"] = std::move(bases);

    llvm::json::Array inputDimensions;
    for (auto [name, size] : layout.getInDims())
      inputDimensions.push_back(
          llvm::json::Object{{"name", name.getValue().str()},
                             {"size", static_cast<int64_t>(size)}});
    object["input_dims"] = std::move(inputDimensions);
    llvm::json::Array outputDimensions;
    for (auto [name, size] : layout.getOutDims())
      outputDimensions.push_back(
          llvm::json::Object{{"name", name.getValue().str()},
                             {"size", static_cast<int64_t>(size)}});
    object["output_dims"] = std::move(outputDimensions);

    llvm::json::Object freeVariables;
    for (auto [name, mask] : layout.getFreeVariableMasks())
      freeVariables[name.getValue()] = static_cast<int64_t>(mask);
    object["free_variable_masks"] = std::move(freeVariables);
    layouts.push_back(std::move(object));
    return id;
  }

  std::string scalarLayout() {
    if (scalarLayoutId)
      return *scalarLayoutId;

    std::string id = (Twine("layout.") + Twine(layouts.size())).str();
    scalarLayoutId = id;
    llvm::json::Object object;
    object["id"] = id;
    object["origin"] = "scalar_semantics";

    llvm::json::Array bases;
    llvm::json::Array inputDimensions;
    llvm::json::Object freeVariables;
    const std::pair<StringRef, StringRef> dimensions[] = {
        {"register", ""},
        {"lane", "ttg.threads-per-warp"},
        {"warp", "ttg.num-warps"},
        {"block", "ttg.num-ctas"},
    };
    for (auto [name, attributeName] : dimensions) {
      int64_t size = 1;
      if (!attributeName.empty()) {
        if (auto attribute = module->getAttrOfType<IntegerAttr>(attributeName))
          size = attribute.getInt();
        else
          addDiagnostic("unsupported.scalar_memory_layout",
                        Twine("missing module attribute ") + attributeName,
                        module.getLoc());
      }
      if (size <= 0 || (size & (size - 1))) {
        addDiagnostic("unsupported.scalar_memory_layout",
                      Twine("non-power-of-two scalar owner dimension ") + name,
                      module.getLoc());
        size = 1;
      }
      llvm::json::Array vectors;
      for (int64_t bits = size; bits > 1; bits >>= 1)
        vectors.push_back(llvm::json::Array{0});
      bases.push_back(llvm::json::Object{{"input", name.str()},
                                         {"basis", std::move(vectors)}});
      inputDimensions.push_back(llvm::json::Object{
          {"name", name.str()}, {"size", static_cast<int64_t>(size)}});
      freeVariables[name] = size - 1;
    }
    object["bases"] = std::move(bases);
    object["input_dims"] = std::move(inputDimensions);
    object["output_dims"] =
        llvm::json::Array{llvm::json::Object{{"name", "dim0"}, {"size", 1}}};
    object["free_variable_masks"] = std::move(freeVariables);
    layouts.push_back(std::move(object));
    return id;
  }

  void collectBaseArguments(Value value, llvm::SetVector<unsigned> &bases,
                            Location useLocation,
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
        if (argument == loop.getInductionVar()) {
          addDiagnostic("unsupported.pointer_provenance",
                        "loop induction variable used as a pointer base",
                        useLocation);
        } else {
          unsigned slot = argument.getArgNumber() - 1;
          collectBaseArguments(loop.getInitArgs()[slot], bases, useLocation,
                               active);
          collectBaseArguments(
              loop.getBody()->getTerminator()->getOperand(slot), bases,
              useLocation, active);
        }
      } else {
        addDiagnostic("unsupported.pointer_provenance",
                      "pointer flows through an unsupported block argument",
                      useLocation);
      }
      active->erase(key);
      return;
    }

    Operation *definition = value.getDefiningOp();
    StringRef name = definition->getName().getStringRef();
    if (name == "tt.addptr" || name == "tt.make_tensor_descriptor" ||
        preservesPointerProvenance(name)) {
      collectBaseArguments(definition->getOperand(0), bases, useLocation,
                           active);
    } else if (name == "arith.select") {
      collectBaseArguments(definition->getOperand(1), bases, useLocation,
                           active);
      collectBaseArguments(definition->getOperand(2), bases, useLocation,
                           active);
    } else if (auto conditional = dyn_cast<scf::IfOp>(definition)) {
      unsigned result = cast<OpResult>(value).getResultNumber();
      collectBaseArguments(conditional.thenYield()->getOperand(result), bases,
                           useLocation, active);
      collectBaseArguments(conditional.elseYield()->getOperand(result), bases,
                           useLocation, active);
    } else if (auto loop = dyn_cast<scf::ForOp>(definition)) {
      unsigned result = cast<OpResult>(value).getResultNumber();
      collectBaseArguments(loop.getInitArgs()[result], bases, useLocation,
                           active);
      collectBaseArguments(loop.getBody()->getTerminator()->getOperand(result),
                           bases, useLocation, active);
    } else {
      addDiagnostic("unsupported.pointer_provenance",
                    Twine("pointer base flows through ") + name, useLocation);
    }
    active->erase(key);
  }

  ModuleOp module;
  ModuleAxisInfoAnalysis axisInfo;
  MLIRContext *context;
  llvm::DenseMap<Operation *, std::string> siteIds;
  llvm::DenseMap<Operation *, std::string> loopIds;
  llvm::DenseMap<unsigned, BlockArgument> functionArguments;
  llvm::DenseMap<Value, int64_t> expressionIds;
  llvm::DenseMap<Value, int64_t> pointerOffsetIds;
  llvm::DenseMap<int64_t, unsigned> expressionDataDepth;
  llvm::DenseMap<int64_t, unsigned> expressionIntegerWidths;
  llvm::DenseMap<Type, std::string> layoutIds;
  std::optional<std::string> scalarLayoutId;
  std::vector<llvm::json::Object> expressions;
  std::vector<llvm::json::Object> layouts;
  std::vector<Diagnostic> diagnostics;
  std::set<std::pair<std::string, std::string>> diagnosticKeys;
  llvm::DenseSet<int64_t> zeroExpressions;
  std::vector<std::string> activeLoops;
  std::vector<std::pair<int64_t, bool>> controlPredicates;
  int64_t nextLoop = 0;
  int64_t lexicalOrder = 0;
};

class AccessManifestPass
    : public PassWrapper<AccessManifestPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AccessManifestPass)

  StringRef getArgument() const final { return "laqs-access-manifest"; }
  StringRef getDescription() const final {
    return "Serialize a launch-specializable Triton memory access manifest";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    ManifestBuilder builder(module);
    llvm::json::Value manifest(builder.build());
    std::string serialized;
    llvm::raw_string_ostream stream(serialized);
    stream << manifest;
    module->setAttr(kManifestAttribute,
                    StringAttr::get(module.getContext(), stream.str()));
  }
};

std::unique_ptr<Pass> createAccessManifestPass() {
  return std::make_unique<AccessManifestPass>();
}

} // namespace
} // namespace mlir::triton::laqs

static void addAccessManifestPass(mlir::PassManager *manager,
                                  const std::vector<std::string> &arguments) {
  if (!arguments.empty())
    llvm::report_fatal_error("laqs_access_manifest pass takes no arguments");
  manager->addPass(mlir::triton::laqs::createAccessManifestPass());
}

static void registerAccessManifestPass() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::triton::laqs::createAccessManifestPass();
  });
}

TRITON_PLUGIN_API mlir::triton::plugin::PluginInfo *tritonGetPluginInfo() {
  static mlir::triton::plugin::PassInfo pass = {"laqs_access_manifest", "1.0.0",
                                                addAccessManifestPass,
                                                registerAccessManifestPass};
  static mlir::triton::plugin::PluginInfo info = {
      TRITON_PLUGIN_API_VERSION,
      "LAQSTritonAccessManifest",
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
