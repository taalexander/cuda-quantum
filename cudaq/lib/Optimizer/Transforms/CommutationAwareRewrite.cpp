/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Transforms/CommutationAwareRewrite.h"
#include "cudaq/Optimizer/Analysis/CommutationAnalysis.h"
#include "cudaq/Optimizer/Analysis/ScalarWireTraversal.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <utility>

using namespace mlir;

namespace {

/// One cursor per virtual qubit in the anchor's support, each walking that
/// qubit's chain of operations. `value` is the wire standing for the qubit
/// where the cursor sits, and `next` is the operation it would reach next.
///
/// Each cursor follows one wire's def-use chain, which is already in program
/// order, so the search is a k-way merge over the frontier rather than a graph
/// traversal: take the earliest head, then advance past it. The frontier never
/// grows, because only the anchor's own qubits matter and a multi-qubit
/// operation joins chains instead of branching into new ones.
struct WireCursor {
  Value value;
  std::optional<cudaq::opt::ScalarWireStep> next;
};

} // namespace

// The search assumes it can tell which qubits an operation touches by looking
// at its operands. That only holds for an operation whose quantum values are
// all scalar wires and that owns no nested code. A reusable control, reference,
// or aggregate operand falls outside the supported form, while a region or
// successor hides code that could reach anything. In each case the operands
// understate what the operation really acts on and the search stops there.
static bool hasBoundedQuantumSupport(Operation *operation) {
  if (operation->getNumRegions() != 0 || operation->getNumSuccessors() != 0)
    return false;
  auto isScalarOrClassical = [](Type type) {
    return !cudaq::quake::isQuantumType(type) ||
           isa<cudaq::quake::WireType>(type);
  };
  return llvm::all_of(operation->getOperandTypes(), isScalarOrClassical) &&
         llvm::all_of(operation->getResultTypes(), isScalarOrClassical);
}

// Order two operations in one supported scope tree.
static bool comesFirst(Operation *lhs, Operation *rhs,
                       const llvm::DenseMap<Operation *, std::size_t> &order) {
  auto lhsPosition = order.find(lhs);
  auto rhsPosition = order.find(rhs);
  if (lhsPosition == order.end() || rhsPosition == order.end())
    return false;
  return lhsPosition->second < rhsPosition->second;
}

// Map one input lane to its result through a validated scalar-wire flow.
static Value mapWireToResult(Operation *operation, Value value) {
  auto flow = cudaq::quake::detail::getScalarWireFlow(operation);
  if (!flow)
    return {};

  for (auto [input, result] : llvm::zip(flow->inputs, flow->results)) {
    if (input == value)
      return result;
  }
  return {};
}

// Scope forwarding is part of the wire path, but it is not a quantum endpoint
// or a crossed effect. Consume those steps until the next direct operation.
static std::optional<cudaq::opt::ScalarWireStep>
getNextDirectStep(Value &wire) {
  while (auto step = cudaq::opt::traverseScalarWire(wire)) {
    if (!step->continueOperand)
      return step;
    wire = step->wire;
  }
  return std::nullopt;
}

// Open one cursor per virtual qubit in the anchor's support.
static llvm::SmallVector<WireCursor>
openFrontier(cudaq::quake::OperatorInterface anchor) {
  llvm::SmallVector<WireCursor> frontier;

  for (Value wire : anchor.getWires()) {
    Value cursorWire = wire;
    frontier.push_back({cursorWire, getNextDirectStep(cursorWire)});
    frontier.back().value = cursorWire;
  }
  return frontier;
}

static void buildOperationOrder(Block &block,
                                llvm::DenseMap<Operation *, std::size_t> &order,
                                std::size_t &nextPosition) {
  for (Operation &operation : block) {
    order.try_emplace(&operation, nextPosition++);
    if (cudaq::opt::isSupportedScalarWireScope(&operation))
      buildOperationOrder(operation.getRegion(0).front(), order, nextPosition);
  }
}

// Take the next operation off the frontier. Every anchor wire must retain an
// unambiguous path in the scope tree. Otherwise the frontier no longer
// represents the anchor's complete support, so the search ends.
static Operation *
takeNext(llvm::ArrayRef<WireCursor> frontier,
         const llvm::DenseMap<Operation *, std::size_t> &operationOrder) {
  Operation *nearest = nullptr;
  for (const WireCursor &cursor : frontier) {
    if (!cursor.next || !operationOrder.contains(cursor.next->operation))
      return nullptr;
    if (!nearest || comesFirst(cursor.next->operation, nearest, operationOrder))
      nearest = cursor.next->operation;
  }
  return nearest;
}

// `QubitIdentityAnalysis` identifies logical qubits, not SSA paths. For
// example, an endpoint can consume a second `borrow_wire` for the same wire-set
// slot while the cursor from the anchor's result reaches another operation.
// The identities match, but the complete frontier reaches the endpoint only
// when every cursor points to it.
static bool doesCompleteFrontierReach(llvm::ArrayRef<WireCursor> frontier,
                                      Operation *candidate) {
  return llvm::all_of(frontier, [candidate](const WireCursor &cursor) {
    return cursor.next && cursor.next->operation == candidate;
  });
}

static bool hasSameOrderedWireTypes(ValueRange lhs, ValueRange rhs) {
  if (lhs.size() != rhs.size())
    return false;
  return llvm::all_of(llvm::zip(lhs, rhs), [](auto pair) {
    auto [lhsValue, rhsValue] = pair;
    return lhsValue.getType() == rhsValue.getType() &&
           isa<cudaq::quake::WireType>(lhsValue.getType());
  });
}

enum class DirectWireThreading { NotDirect, Exact, Mismatch };

// A unary operation cannot repeat a qubit in another operand role. Multi-wire
// operations need identity normalization to establish that precondition.
static bool requiresDistinctQubitProof(Operation *operation) {
  auto operatorInterface = cast<cudaq::quake::OperatorInterface>(operation);
  return cudaq::quake::getWireOperands(operatorInterface).size() > 1;
}

// With no crossed operation, exact result-to-operand threading proves the
// endpoint pair's ordered SSA correspondence. The caller still owns the
// endpoint algebra, and identity analysis still validates operand uniqueness
// for multi-wire operations.
//
// A direct pair that permutes values or changes control and target roles is a
// definitive mismatch. Falling back to logical identity matching could hide
// that mismatch by resolving different SSA values to the same qubit.
static DirectWireThreading classifyDirectWireThreading(
    Operation *lhs, Operation *rhs,
    const llvm::DenseMap<Operation *, std::size_t> &operationOrder) {
  auto lhsInterface = dyn_cast<cudaq::quake::OperatorInterface>(lhs);
  auto rhsInterface = dyn_cast<cudaq::quake::OperatorInterface>(rhs);
  if (!lhsInterface || !rhsInterface)
    return DirectWireThreading::NotDirect;

  auto lhsPosition = operationOrder.find(lhs);
  auto rhsPosition = operationOrder.find(rhs);
  if (lhsPosition == operationOrder.end() ||
      rhsPosition == operationOrder.end())
    return DirectWireThreading::NotDirect;
  bool lhsIsProducer = lhsPosition->second < rhsPosition->second;
  ValueRange producerResults =
      lhsIsProducer ? lhsInterface.getWires() : rhsInterface.getWires();
  auto consumerOperands = cudaq::quake::getWireOperands(
      lhsIsProducer ? rhsInterface : lhsInterface);
  if (!hasSameOrderedWireTypes(producerResults, consumerOperands))
    return DirectWireThreading::NotDirect;

  llvm::SmallVector<Value> reachedOperands;
  for (Value result : producerResults) {
    Value wire = result;
    auto step = getNextDirectStep(wire);
    if (!step || step->operation != (lhsIsProducer ? rhs : lhs))
      return DirectWireThreading::NotDirect;
    reachedOperands.push_back(wire);
  }

  bool hasSameRoles = hasSameOrderedWireTypes(lhsInterface.getControls(),
                                              rhsInterface.getControls()) &&
                      hasSameOrderedWireTypes(lhsInterface.getTargets(),
                                              rhsInterface.getTargets());
  bool hasSameValues =
      llvm::all_of(llvm::zip(reachedOperands, consumerOperands), [](auto pair) {
        auto [reached, consumer] = pair;
        return reached == consumer;
      });
  if (hasSameValues)
    return hasSameRoles ? DirectWireThreading::Exact
                        : DirectWireThreading::Mismatch;

  llvm::DenseSet<Value> producerValues(reachedOperands.begin(),
                                       reachedOperands.end());
  if (llvm::all_of(consumerOperands, [&](Value operand) {
        return producerValues.contains(operand);
      }))
    return DirectWireThreading::Mismatch;
  return DirectWireThreading::NotDirect;
}

// Step past `candidate`. An operation on several of the anchor's qubits is the
// head of several chains at once, so every cursor pointing at it moves on
// together and the operation is visited once rather than once per qubit. A
// cursor whose qubit does not continue past `candidate`, such as one reaching a
// sink, a returned wire, or a block argument, simply ends.
static void advanceFrontierPast(llvm::SmallVectorImpl<WireCursor> &frontier,
                                Operation *candidate) {
  for (WireCursor &cursor : frontier) {
    if (!cursor.next || cursor.next->operation != candidate)
      continue;
    Value stepped = mapWireToResult(candidate, cursor.value);
    cursor.value = stepped;
    cursor.next = stepped ? getNextDirectStep(cursor.value) : std::nullopt;
  }
}

class cudaq::opt::CommutationAwareRewriteMatcher::Impl {
public:
  // Build one analysis for the connected ordinary-scope tree. Reconstructing
  // discarded state is also accounted as a fallback rebuild.
  cudaq::quake::detail::CommutationAnalysis &getAnalysis(Block *block) {
    Block *root = cudaq::opt::getScalarWireTraversalRoot(block);
    auto [entry, inserted] = analyses.try_emplace(root);
    if (inserted) {
      entry->second =
          std::make_unique<cudaq::quake::detail::CommutationAnalysis>(*root);
      ++statistics.analysisBuilds;
      if (invalidatedBlocks.erase(root))
        ++statistics.fallbackRebuilds;
    }
    return *entry->second;
  }

  auto findAnalysis(Block *block) {
    return analyses.find(cudaq::opt::getScalarWireTraversalRoot(block));
  }

  const llvm::DenseMap<Operation *, std::size_t> &
  getOperationOrder(Block *root) {
    auto [entry, inserted] = operationOrders.try_emplace(root);
    if (inserted) {
      std::size_t nextPosition = 0;
      buildOperationOrder(*root, entry->second, nextPosition);
    }
    return entry->second;
  }

  static bool isNestedIn(Block *block, Block *ancestor) {
    while (block) {
      if (block == ancestor)
        return true;
      Operation *parent = block->getParentOp();
      block = parent ? parent->getBlock() : nullptr;
    }
    return false;
  }

  void discardOperationOrder(Block *block) {
    if (!block)
      return;
    llvm::SmallVector<Block *> affectedRoots;
    for (auto &entry : operationOrders)
      if (isNestedIn(block, entry.first))
        affectedRoots.push_back(entry.first);
    for (Block *root : affectedRoots)
      operationOrders.erase(root);
  }

  // A nested mutation can change whether an enclosing scope is traversable,
  // so discard every live analysis rooted above it.
  void discardBlock(Block *block) {
    if (!block)
      return;
    llvm::SmallVector<Block *> affectedRoots;
    for (auto &entry : analyses) {
      Block *root = entry.first;
      if (isNestedIn(block, root))
        affectedRoots.push_back(root);
    }
    for (Block *root : affectedRoots) {
      invalidatedBlocks.insert(root);
      operationOrders.erase(root);
      analyses.erase(root);
    }
  }

  // Analyses rooted in an erased subtree cannot be rebuilt. An analysis above
  // that subtree remains live code and records fallback if queried again.
  void forgetBlock(Block *block) {
    if (!block)
      return;
    llvm::SmallVector<std::pair<Block *, bool>> affectedRoots;
    for (auto &entry : analyses) {
      Block *root = entry.first;
      if (isNestedIn(root, block))
        affectedRoots.emplace_back(root, false);
      else if (isNestedIn(block, root))
        affectedRoots.emplace_back(root, true);
    }
    for (auto [root, canRebuild] : affectedRoots) {
      if (canRebuild)
        invalidatedBlocks.insert(root);
      else
        invalidatedBlocks.erase(root);
      operationOrders.erase(root);
      analyses.erase(root);
    }
  }

  llvm::DenseMap<Block *,
                 std::unique_ptr<cudaq::quake::detail::CommutationAnalysis>>
      analyses;
  llvm::DenseSet<Block *> invalidatedBlocks;
  llvm::DenseMap<Block *, llvm::DenseMap<Operation *, std::size_t>>
      operationOrders;
  cudaq::opt::CommutationAwareRewriteStatistics statistics;
};

namespace cudaq::opt::detail {
class CommutationAwareRewriteListener
    : public RewriterBase::ForwardingListener {
public:
  CommutationAwareRewriteListener(CommutationAwareRewriteMatcher &matcher,
                                  OpBuilder::Listener *listener)
      : RewriterBase::ForwardingListener(listener), matcher(matcher) {}

  void notifyOperationInserted(Operation *operation,
                               OpBuilder::InsertPoint previous) override {
    RewriterBase::ForwardingListener::notifyOperationInserted(operation,
                                                              previous);
    matcher.impl->discardOperationOrder(operation->getBlock());
    // A previous insertion point identifies a move. Invalidate cached state
    // for both the source and destination blocks.
    if (Block *previousBlock = previous.getBlock()) {
      matcher.impl->discardOperationOrder(previousBlock);
      discardBlock(previousBlock);
      if (operation->getBlock() != previousBlock)
        discardBlock(operation->getBlock());
      return;
    }

    if (operation->getNumRegions() != 0 || operation->getNumSuccessors() != 0 ||
        operation->hasTrait<OpTrait::IsTerminator>() ||
        isa<cudaq::cc::UnwindBreakOp, cudaq::cc::UnwindContinueOp,
            cudaq::cc::UnwindReturnOp>(operation)) {
      discardBlock(operation->getBlock());
      return;
    }

    // A new operation can be maintained incrementally only when its result
    // identities propagate unambiguously in the current block.
    auto analysis = matcher.impl->findAnalysis(operation->getBlock());
    if (analysis != matcher.impl->analyses.end() &&
        !analysis->second->registerIdentityPreservingOperation(operation))
      discardBlock(operation->getBlock());
  }

  void notifyBlockInserted(Block *block, Region *previous,
                           Region::iterator previousIt) override {
    RewriterBase::ForwardingListener::notifyBlockInserted(block, previous,
                                                          previousIt);
    matcher.impl->discardOperationOrder(block);
    if (previous && previous->getParentOp()) {
      matcher.impl->discardOperationOrder(previous->getParentOp()->getBlock());
      discardBlock(previous->getParentOp()->getBlock());
    }
    discardBlock(block);
  }

  void notifyBlockErased(Block *block) override {
    RewriterBase::ForwardingListener::notifyBlockErased(block);
    matcher.impl->discardOperationOrder(block);
    matcher.impl->forgetBlock(block);
  }

  void notifyOperationModified(Operation *operation) override {
    RewriterBase::ForwardingListener::notifyOperationModified(operation);
    // A use rewired by a validated identity-preserving replacement changes the
    // operand value but not the virtual qubit it denotes, and every commutation
    // rule reads only identities, roles, polarity, and semantics. Such a user
    // therefore needs no invalidation; the notification is consumed only so it
    // is not mistaken for an unexplained in-place change.
    auto pending = pendingIdentityPreservingUsers.find(operation);
    if (pending != pendingIdentityPreservingUsers.end()) {
      assert(pending->second != 0 && "pending replacement count underflow");
      if (--pending->second == 0)
        pendingIdentityPreservingUsers.erase(pending);
      return;
    }
    // Any other in-place change may alter commutation semantics or identity
    // placement, so rebuild the affected block conservatively.
    discardBlock(operation->getBlock());
  }

  void notifyOperationReplaced(Operation *operation,
                               Operation *replacement) override {
    RewriterBase::ForwardingListener::notifyOperationReplaced(operation,
                                                              replacement);
    updateReplacement(operation, replacement->getResults());
  }

  void notifyOperationReplaced(Operation *operation,
                               ValueRange replacement) override {
    RewriterBase::ForwardingListener::notifyOperationReplaced(operation,
                                                              replacement);
    updateReplacement(operation, replacement);
  }

  void notifyOperationErased(Operation *operation) override {
    RewriterBase::ForwardingListener::notifyOperationErased(operation);
    matcher.impl->discardOperationOrder(operation->getBlock());
    pendingIdentityPreservingUsers.erase(operation);
    if (operation->getNumRegions() != 0 || operation->getNumSuccessors() != 0 ||
        operation->hasTrait<OpTrait::IsTerminator>() ||
        isa<cudaq::cc::UnwindBreakOp, cudaq::cc::UnwindContinueOp,
            cudaq::cc::UnwindReturnOp>(operation)) {
      discardBlock(operation->getBlock());
      return;
    }
    auto analysis = matcher.impl->findAnalysis(operation->getBlock());
    if (analysis == matcher.impl->analyses.end())
      return;
    analysis->second->eraseOperation(operation);
  }

private:
  void updateReplacement(Operation *operation, ValueRange replacement) {
    // Replacement callbacks and their per-use modification callbacks are
    // synchronous. Starting another replacement or falling back must never
    // leave counts that could suppress a later genuine modification.
    pendingIdentityPreservingUsers.clear();
    auto analysis = matcher.impl->findAnalysis(operation->getBlock());
    if (analysis == matcher.impl->analyses.end())
      return;

    // Quantum rewiring is incrementally maintainable only after identity
    // validation. A used classical result can feed an operator parameter or
    // Pauli value, so its user semantics may change even when every quantum
    // identity is unchanged.
    for (Value result : operation->getResults()) {
      if (!result.use_empty() &&
          !cudaq::quake::isQuantumType(result.getType())) {
        discardBlock(operation->getBlock());
        return;
      }
    }

    // A validated replacement preserves qubit identity state and invalidates
    // only pairs incident to the replaced endpoint. Failure requires block
    // fallback.
    if (!analysis->second->prepareIdentityPreservingReplacement(operation,
                                                                replacement)) {
      discardBlock(operation->getBlock());
      return;
    }

    // RewriterBase emits exactly one modification notification per replaced
    // use. Count quantum uses per owner so a multi-result replacement into one
    // user consumes every callback without suppressing anything afterward.
    for (Value result : operation->getResults())
      for (OpOperand &use : result.getUses())
        ++pendingIdentityPreservingUsers[use.getOwner()];
  }

  void discardBlock(Block *block) {
    pendingIdentityPreservingUsers.clear();
    matcher.impl->discardBlock(block);
  }

  CommutationAwareRewriteMatcher &matcher;
  llvm::DenseMap<Operation *, std::size_t> pendingIdentityPreservingUsers;
};
} // namespace cudaq::opt::detail

class cudaq::opt::CommutationAwareRewriteDriver::Impl {
public:
  Impl(MLIRContext &context, GreedyRewriteConfig config,
       std::unique_ptr<CommutationAwareRewriteMatcher> matcher)
      : matcher(std::move(matcher)), patterns(&context),
        config(std::move(config)),
        listener(*this->matcher, this->config.getListener()) {
    this->config.setRegionSimplificationLevel(
        GreedySimplifyRegionLevel::Disabled);
    this->config.setListener(&listener);
  }

  std::unique_ptr<CommutationAwareRewriteMatcher> matcher;
  RewritePatternSet patterns;
  GreedyRewriteConfig config;
  cudaq::opt::detail::CommutationAwareRewriteListener listener;
  bool hasRun = false;
};

cudaq::opt::CommutationAwareRewriteMatcher::CommutationAwareRewriteMatcher(
    std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

cudaq::opt::CommutationAwareRewriteMatcher::~CommutationAwareRewriteMatcher() =
    default;

Operation *cudaq::opt::CommutationAwareRewriteMatcher::findNearest(
    Operation *anchor, llvm::function_ref<bool(Operation *)> isEndpoint) {
  if (!anchor || !anchor->getBlock())
    return nullptr;
  auto anchorInterface = dyn_cast<cudaq::quake::OperatorInterface>(anchor);
  if (!anchorInterface)
    return nullptr;

  Block *block = anchor->getBlock();
  Block *root = cudaq::opt::getScalarWireTraversalRoot(block);
  if (!hasBoundedQuantumSupport(anchor))
    return nullptr;

  const auto &operationOrder = impl->getOperationOrder(root);

  // Follow Quake's own wire dataflow rather than block order. Only
  // operations sharing a virtual qubit with the anchor are reachable this way.
  // Every operation skipped is disjoint from the anchor's support and therefore
  // commutes with it, so it needs neither a probe nor a cache entry.
  auto frontier = openFrontier(anchorInterface);

  cudaq::quake::detail::CommutationAnalysis *analysis = nullptr;
  auto requireAnalysis = [&]() -> cudaq::quake::detail::CommutationAnalysis & {
    if (!analysis)
      analysis = &impl->getAnalysis(block);
    return *analysis;
  };
  // A self-query builds the normalized operation view and rejects a logical
  // qubit used in more than one role. Unary operations cannot violate that
  // constraint and retain the analysis-free adjacent path.
  auto hasDistinctQubits = [&](Operation *operation) {
    return !requiresDistinctQubitProof(operation) ||
           requireAnalysis().canCommute(operation, operation);
  };
  while (Operation *candidate = takeNext(frontier, operationOrder)) {
    // Reference and aggregate quantum values, and nested code that could reach
    // further qubits, are outside the adopted semantics. Measurement
    // instruments and reset channels are the only non-unitary operations with
    // supported scalar-wire flow; all other reached operations retain the
    // conservative self-query barrier.
    if (!hasBoundedQuantumSupport(candidate))
      return nullptr;
    bool isTraversableMeasurementOrReset =
        isa<cudaq::quake::MeasurementInterface, cudaq::quake::ResetOp>(
            candidate);
    auto candidateInterface =
        dyn_cast<cudaq::quake::OperatorInterface>(candidate);

    // Consumer policy decides endpoint compatibility. Every operation crossed
    // before an accepted endpoint requires the commutation proof below.
    if (candidateInterface &&
        (!hasDistinctQubits(anchor) || !hasDistinctQubits(candidate)))
      return nullptr;
    if (candidateInterface && isEndpoint(candidate)) {
      if (!doesCompleteFrontierReach(frontier, candidate))
        return nullptr;
      return candidate;
    }

    // A candidate that is crossed rather than accepted must be proven to
    // commute with the anchor.
    auto &blockAnalysis = requireAnalysis();
    // The anchor and candidate must be resolvable before any pair result
    // involving them means anything. Measurement instruments and reset
    // channels are traversable through their scalar-wire flow, but other
    // candidates keep the conservative self-query barrier.
    if (!blockAnalysis.canCommute(anchor, anchor) ||
        (!isTraversableMeasurementOrReset &&
         !blockAnalysis.canCommute(candidate, candidate)))
      return nullptr;
    if (!blockAnalysis.canCommute(anchor, candidate))
      return nullptr;

    advanceFrontierPast(frontier, candidate);
  }
  return nullptr;
}

bool cudaq::opt::CommutationAwareRewriteMatcher::haveSameOrderedQuantumOperands(
    Operation *lhs, Operation *rhs) {
  if (!lhs || !rhs || !lhs->getBlock() || !rhs->getBlock())
    return false;
  Block *root = cudaq::opt::getScalarWireTraversalRoot(lhs->getBlock());
  if (cudaq::opt::getScalarWireTraversalRoot(rhs->getBlock()) != root)
    return false;
  const auto &operationOrder = impl->getOperationOrder(root);
  auto directThreading = classifyDirectWireThreading(lhs, rhs, operationOrder);
  if (directThreading == DirectWireThreading::Exact) {
    // Exact threading proves ordered operands for unary endpoints. Multi-wire
    // endpoints still need normalized views that reject duplicate qubit roles.
    if (!requiresDistinctQubitProof(lhs) && !requiresDistinctQubitProof(rhs))
      return true;
    auto &analysis = impl->getAnalysis(lhs->getBlock());
    return analysis.canCommute(lhs, lhs) && analysis.canCommute(rhs, rhs);
  }
  if (directThreading == DirectWireThreading::Mismatch)
    return false;
  return impl->getAnalysis(lhs->getBlock())
      .haveSameOrderedQuantumOperands(lhs, rhs);
}

cudaq::opt::CommutationAwareRewriteDriver::CommutationAwareRewriteDriver(
    MLIRContext &context, GreedyRewriteConfig config)
    : impl(std::make_unique<Impl>(
          context, std::move(config),
          std::unique_ptr<CommutationAwareRewriteMatcher>(
              new CommutationAwareRewriteMatcher(
                  std::make_unique<CommutationAwareRewriteMatcher::Impl>())))) {
}

cudaq::opt::CommutationAwareRewriteDriver::~CommutationAwareRewriteDriver() =
    default;

RewritePatternSet &cudaq::opt::CommutationAwareRewriteDriver::getPatterns() {
  return impl->patterns;
}

cudaq::opt::CommutationAwareRewriteMatcher &
cudaq::opt::CommutationAwareRewriteDriver::getMatcher() {
  return *impl->matcher;
}

cudaq::opt::CommutationAwareRewriteStatistics
cudaq::opt::CommutationAwareRewriteDriver::getStatistics() const {
  return impl->matcher->impl->statistics;
}

LogicalResult cudaq::opt::CommutationAwareRewriteDriver::run(Region &region) {
  if (impl->hasRun)
    return failure();
  impl->hasRun = true;
  return applyPatternsGreedily(region, std::move(impl->patterns), impl->config);
}
