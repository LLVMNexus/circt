// RUN: circt-opt %s --export-chisel-interface | FileCheck %s

// CHECK-LABEL: // Generated by CIRCT
// CHECK-LABEL: package shelf.foo
// CHECK-LABEL: import chisel3._
// CHECK-NEXT: import chisel3.experimental._

// CHECK-LABEL: class Foo extends ExtModule {
// CHECK-NEXT:    val reset = IO(Input(AsyncReset()))
// CHECK-NEXT:    val clock = IO(Input(Clock()))
// CHECK-NEXT:    val uInt = IO(Input(UInt(2.W)))
// CHECK-NEXT:    val sInt = IO(Output(SInt(4.W)))
// CHECK-NEXT:    val analog = IO(Analog(5.W))
// CHECK-NEXT:    val bundle = IO(new Bundle {
// CHECK-NEXT:      val a = Input(UInt(32.W))
// CHECK-NEXT:      val b = Analog(1.W)
// CHECK-NEXT:      val c = Input(new Bundle {
// CHECK-NEXT:        val d = SInt(2.W)
// CHECK-NEXT:        val e = SInt(4.W)
// CHECK-NEXT:      })
// CHECK-NEXT:    })
// CHECK-NEXT:    val bundleWithFlip = IO(new Bundle {
// CHECK-NEXT:      val word = Output(UInt(32.W))
// CHECK-NEXT:      val valid = Output(UInt(1.W))
// CHECK-NEXT:      val ready = Input(UInt(1.W))
// CHECK-NEXT:    })
// CHECK-NEXT:    val vector = IO(Input(Vec(10, UInt(16.W))))
// CHECK-NEXT:    val vectorOfVector = IO(Input(Vec(20, Vec(10, UInt(16.W)))))
// CHECK-NEXT:    val vectorOfBundle = IO(Vec(5, new Bundle {
// CHECK-NEXT:      val word = Input(UInt(32.W))
// CHECK-NEXT:      val valid = Input(UInt(1.W))
// CHECK-NEXT:      val ready = Output(UInt(1.W))
// CHECK-NEXT:    }))
// CHECK-NEXT:    val constUInt = IO(Input(Const(UInt(3.W))))
// CHECK-NEXT:    val constBundle = IO(Input(Const(new Bundle {
// CHECK-NEXT:      val a = SInt(8.W)
// CHECK-NEXT:    })))
// CHECK-NEXT:    val mixedConstBundle = IO(Input(new Bundle {
// CHECK-NEXT:      val a = SInt(8.W)
// CHECK-NEXT:      val b = Const(UInt(4.W))
// CHECK-NEXT:    }))
// CHECK-NEXT:  }
firrtl.circuit "Foo"  {
    firrtl.module @Foo(
      in %reset: !firrtl.asyncreset,
      in %clock: !firrtl.clock,
      in %uInt: !firrtl.uint<2>,
      out %sInt: !firrtl.sint<4>,
      out %analog: !firrtl.analog<5>,
      in %bundle: !firrtl.bundle<a: uint<32>, b: analog<1>, c: bundle<d: sint<2>, e: sint<4>>>,
      out %bundleWithFlip: !firrtl.bundle<word: uint<32>, valid: uint<1>, ready flip: uint<1>>,
      in %vector: !firrtl.vector<uint<16>, 10>,
      in %vectorOfVector: !firrtl.vector<vector<uint<16>, 10>, 20>,
      in %vectorOfBundle: !firrtl.vector<bundle<word: uint<32>, valid: uint<1>, ready flip: uint<1>>, 5>,
      in %constUInt: !firrtl.const.uint<3>,
      in %constBundle: !firrtl.const.bundle<a: sint<8>>,
      in %mixedConstBundle: !firrtl.bundle<a: sint<8>, b: const.uint<4>>) {}
}
