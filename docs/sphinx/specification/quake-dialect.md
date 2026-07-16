# Quake Dialect

## General Introduction

The quantum circuit model is the most widely used model of quantum
computation.  It provides a convenient tool for formulating quantum
algorithms and an architecture for the physical construction of quantum
computers.

A _quantum circuit_ represents a computation as a sequence of quantum
operators applied to quantum data.  In our case, the quantum data is a set
of quantum bits, or qubits for short.  Physically, a qubit is an object with
only two distinguishable states, i.e., it is a two-state quantum mechanical
system such as a spin-1/2 particle.

Conceptually, a _quantum operator_ is an effect that might modify the state
of a subset of qubits. Most often, this effect is unitary evolution. In this
case, we say that the operator is a _unitary_.  The number of target
qubits an operator acts upon is an intrinsic property.

A _quantum instruction_ is the embodiment of a quantum operator when applied
to a specific subset of qubits.  The number of qubits must be equal to (or
greater than) the number of target qubits intrinsic to the operator.  If
greater, the extra qubits are considered controls.

## Motivation

The main motivation behind Quake's value model is to directly expose
quantum and classical data dependencies for optimization purposes,
i.e., to represent the dataflow in quantum computations.  In contrast
to Quake's memory model, which uses memory semantics (quantum
operators act as side-effects on qubit references), the value model
uses value semantics, that is quantum operators consume and produce
values. These values are not truly SSA values, however, as operations
still have side-effects on the value itself and the value cannot be
copied.

The following Quake fragment illustrates the distinction between the models:

```mlir
func.func @foo(%veq : !quake.veq<2>) {
    // Boilerplate to extract each qubit from the vector
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %q0 = quake.extract_ref %veq[%c0] : (!quake.veq<2>, index) -> !quake.ref
    %q1 = quake.extract_ref %veq[%c1] : (!quake.veq<2>, index) -> !quake.ref

    // We apply some operators to those extracted qubits
    // ... bunch of operators using %q0 and %q1 ...
    quake.h %q0 : (!quake.ref) -> ()

    // We decide to measure the vector
    %result = quake.mz %veq : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>

    // And then apply another Hadamard to %q0
    quake.h %q0 : (!quake.ref) -> ()
    return
}
```

Consider an optimization that removes inverse operations that are adjacent in a
qubit's direct use chain. A pair of Hadamard operations is one example:

```text
    ┌───┐ ┌───┐         ┌───┐
   ─┤ H ├─┤ H ├─  =  ───┤ I ├───  =  ─────────────
    └───┘ └───┘         └───┘
```

`I` is the identity operator. An implementation that only compares the two
`quake.h` operands could remove both operations. It would miss the measurement
of `%veq`, which aliases `%q0` through `quake.extract_ref`.

A correct implementation can account for aliases between a vector and its
extracted references, but doing so requires additional analysis. Quake's value
form makes these dependencies explicit.

In the value model operators consume values and return new values:

```text
  %q0_1 = quake.op %q0_0 : (!quake.wire) -> !quake.wire
```

The memory and value representations differ as follows:

```text
            Memory                                   Value

        ┌──┐ ┌──┐     ┌──┐                  ┌──┐ %q0_1 ┌──┐     ┌──┐
   %q0 ─┤  ├─┤  ├─···─┤  ├─ %q0  vs  %q0_0 ─┤  ├───────┤  ├─···─┤  ├─ %q0_Z
        └──┘ └──┘     └──┘                  └──┘       └──┘     └──┘
```

In the reference form, both Hadamard operations use `%q0`, while the
measurement reaches that reference through `%veq`. Quake's intermediate quantum
load/store form exposes wire segments between `quake.unwrap` and
`quake.wrap` while retaining reference storage at their boundaries. The
corresponding mixed reference and wire fragment is:

```text
func.func @foo(%array : !quake.veq<2>) {
    // Boilerplate to extract each qubit
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %r0 = quake.extract_ref %array[%c0] : (!quake.veq<2>, index) -> !quake.ref
    %r1 = quake.extract_ref %array[%c1] : (!quake.veq<2>, index) -> !quake.ref

    // Unwrap the quantum references to expose the wires.
    %q0 = quake.unwrap %r0 : (!quake.ref) -> !quake.wire
    %q1 = quake.unwrap %r1 : (!quake.ref) -> !quake.wire

    // Misc. operators applied
    %q0_M = quake.h %q0_L : (!quake.wire) -> !quake.wire

    // Re-wrap the wire to its original source
    quake.wrap %q0_M to %r0 : !quake.wire, !quake.ref
    quake.wrap %q1_X to %r1 : !quake.wire, !quake.ref

    // Measure the entire vector of quantum references
    %result = quake.mz %array : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>

    // Unwrap the wire for qubit 0 again
    %q0_P = quake.unwrap %r0 : (!quake.ref) -> !quake.wire
    ...
    %q0_Z = quake.h %q0_Y : (!quake.wire) -> !quake.wire
    // Re-wrap the wire back to the original reference
    quake.wrap %q0_Z to %r0 : !quake.wire, !quake.ref
    return
}
```

The Hadamard operations now belong to separate wire chains in this form:

```text
Memory                          Value
    %q0         [%q0_0, %q0_1 ... %q0_L, %q0_M; %q0_P ... %q0_Y, %q0_Z]

```

One Hadamard consumes `%q0_L` and produces `%q0_M`. The other consumes
`%q0_Y` and produces `%q0_Z`. The measurement separates those chains, so the
two operations cannot cancel.
