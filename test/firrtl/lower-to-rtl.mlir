// RUN: cirt-opt -lower-firrtl-to-rtl %s | FileCheck %s

// CHECK-LABEL: firrtl.module @Constant
firrtl.module @Constant(%in1: !firrtl.uint<4>,
                        %in2: !firrtl.uint<2>,
                        %out1: !firrtl.flip<uint<4>>) {

  // CHECK: rtl.constant(-4 : i4) : i4
  %c12_ui4 = firrtl.constant(12 : ui4) : !firrtl.uint<4>
 
  // CHECK: rtl.constant(2 : i3) : i3
  %c2_si3 = firrtl.constant(2 : si3) : !firrtl.sint<3>

  // CHECK: %0 = rtl.add %c-4_i4, %c-4_i4 : i4
  %0 = firrtl.add %c12_ui4, %c12_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

  // CHECK: %1 = firrtl.stdIntCast %in1 : (!firrtl.uint<4>) -> i4
  %1 = firrtl.asUInt %in1 : (!firrtl.uint<4>) -> !firrtl.uint<4>

  // CHECK: %2 = rtl.sub %0, %1 : i4
  %2 = firrtl.sub %0, %1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

  // CHECK: %3 = firrtl.stdIntCast %in2 : (!firrtl.uint<2>) -> i2
  // CHECK: %4 = rtl.sext %3 : i2, i3
  %3 = firrtl.pad %in2, 3 : (!firrtl.uint<2>) -> !firrtl.sint<3>

  // CHECK: %5 = rtl.zext %4 : i3, i4
  %4 = firrtl.pad %3, 4 : (!firrtl.sint<3>) -> !firrtl.uint<4>

  // CHECK: %6 = rtl.xor %2, %5 : i4
  %5 = firrtl.xor %2, %4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

  // CHECK-NEXT: firrtl.connect %out1, %6 : !firrtl.flip<uint<4>>, i4
  firrtl.connect %out1, %5 : !firrtl.flip<uint<4>>, !firrtl.uint<4>
}
