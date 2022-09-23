// RUN: circt-dis --help | FileCheck %s --implicit-check-not='{{[Oo]}}ptions:'

// CHECK: OVERVIEW: CIRCT .mlirbc -> .mlir disassembler
// CHECK: Color Options
// CHECK: General {{[Oo]}}ptions
// CHECK-NOT: --{{[^m][^l][^i][^r]}}-
// CHECK: Generic Options
// CHECK: circt-dis Options
