// RUN: circt-opt --lower-sim-to-sv %s | FileCheck %s

sim.func.dpi @dpi1(in %arg1: i1)
sim.func.dpi @dpi2(in %arg1: i1)
// CHECK:       sv.func private @dpi1(in %arg1 : i1)
// CHECK-NEXT:  emit.fragment @dpi1_dpi_import_fragment {
// CHECK-NEXT:    sv.func.dpi.import @dpi1
// CHECK-NEXT:  }
// CHECK-NEXT:  sv.func private @dpi2(in %arg1 : i1)
// CHECK-NEXT:  emit.fragment @dpi2_dpi_import_fragment {
// CHECK-NEXT:    sv.func.dpi.import @dpi2
// CHECK-NEXT:  }

// CHECK-LABEL: hw.module @dpi_call1
// CHECK-SAME: {emit.fragments = [@dpi1_dpi_import_fragment, @dpi2_dpi_import_fragment]}
hw.module @dpi_call1(in %clock : !seq.clock, in %enable : i1, in %in: i1) {
  sim.func.dpi.call @dpi1(%in) clock %clock enable %enable: (i1) -> ()
  // CHECK: %[[CLK:.+]] = seq.from_clock %clock
  // CHECK-NEXT: sv.always posedge %[[CLK]] {
  // CHECK-NEXT:     sv.if %enable {
  // CHECK-NEXT:       sv.func.call.procedural @dpi1(%in) : (i1) -> ()
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }

  sim.func.dpi.call @dpi2(%in) clock %clock : (i1) -> ()
  // CHECK: %[[CLK:.+]] = seq.from_clock %clock
  // CHECK-NEXT: sv.always posedge %[[CLK]] {
  // CHECK-NEXT:   sv.func.call.procedural @dpi2(%in) : (i1) -> () 
  // CHECK-NEXT: }
}

// CHECK-LABEL: hw.module @dpi_call2
// CHECK-SAME: {emit.fragments = [@dpi1_dpi_import_fragment]}
hw.module @dpi_call2(in %clock : !seq.clock, in %enable : i1, in %in: i1) {
  sim.func.dpi.call @dpi1(%in) clock %clock enable %enable: (i1) -> ()
  // CHECK: %[[CLK:.+]] = seq.from_clock %clock
  // CHECK-NEXT: sv.always posedge %[[CLK]] {
  // CHECK-NEXT:     sv.if %enable {
  // CHECK-NEXT:       sv.func.call.procedural @dpi1(%in) : (i1) -> ()
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
}