; RUN: firtool %s --format=fir  | FileCheck %s

; CHECK-LABEL: `ifndef SYNTHESIS
; CHECK-NEXT:  import "DPI-C" function void foo(
; CHECK-NEXT:    input  [1:0] in_0,
; CHECK-NEXT:                 in_1,
; CHECK-NEXT:    output  struct packed {logic [31:0] out1; logic out2; } out_0
; CHECK-NEXT:  );
circuit DPI:
  intmodule DPICall :
    input clock : Clock
    input in1: UInt<2>
    input in2: UInt<2>
    output out : {out1: UInt<32>, out2: UInt<1>}
    intrinsic = circt_dpi_import_and_call
    parameter function_name = "foo"
  
  ; CHECK-LABEL: module DPI(
  ; CHECK:      struct packed {logic [31:0] out1; logic out2; } _GEN;
  ; CHECK-NEXT: always @(posedge clock) begin
  ; CHECK-NEXT:   foo(in_0, in_1, _GEN);
  ; CHECK-NEXT:   if (_GEN.out2)
  ; CHECK-NEXT:     assert(cond) else $error("baz");
  ; CHECK-NEXT: end // always @(posedge)
  ; CHECK-NEXT: assign out1 = {31'h0, _GEN.out2};
  module DPI :
    input clock: Clock
    input cond: UInt<1>
    input in: UInt<2>[2]
    output out1 : UInt<32>


    inst dpi of DPICall 
    dpi.in1 <= in[0]
    dpi.in2 <= in[1]
    dpi.clock <= clock
    out1 <= dpi.out.out2
    assert(clock, cond, dpi.out.out2 "baz")