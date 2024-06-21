// RUN: circt-opt --moore-simplify-assigns %s | FileCheck %s

 moore.module @Foo() {
    %a = moore.variable : <i8960>
    %b = moore.variable : <i42>
    %c = moore.variable : <i9002>
    %u = moore.variable : <l8960>
    %v = moore.variable : <l42>
    %w = moore.variable : <l9002>
    // CHECK-NOT: moore.concat_ref
    %0 = moore.concat_ref %a, %b : (!moore.ref<i8960>, !moore.ref<i42>) -> <i9002>
    // CHECK: %[[C_READ:.+]] = moore.read %c : i9002
    %1 = moore.read %c : i9002
    // CHECK: %[[CONST_42:.+]] = moore.constant 42 : i32
    // CHECK: %[[TMP1:.+]] = moore.extract %[[C_READ]] from %[[CONST_42]] : i9002, i32 -> i8960
    // CHECK: moore.assign %a, %[[TMP1]] : i8960
    // CHECK: %[[CONST_0:.+]] = moore.constant 0 : i32
    // CHECK: %[[TMP2:.+]] = moore.extract %[[C_READ]] from %[[CONST_0]] : i9002, i32 -> i42
    // CHECK: moore.assign %b, %[[TMP2]] : i42
    moore.assign %0, %1 : i9002
    moore.procedure always {
      // CHECK-NOT: moore.concat_ref
      %2 = moore.concat_ref %u, %v : (!moore.ref<l8960>, !moore.ref<l42>) -> <l9002>
      // CHECK: %[[W_READ:.+]] = moore.read %w : l9002
      // CHECK: %[[CONST_42:.+]] = moore.constant 42 : i32
      // CHECK: %[[TMP1:.+]] = moore.extract %[[W_READ]] from %[[CONST_42]] : l9002, i32 -> l8960
      // CHECK: moore.blocking_assign %u, %[[TMP1]] : l8960
      // CHECK: %[[CONST_0:.+]] = moore.constant 0 : i32
      // CHECK: %[[TMP2:.+]] = moore.extract %[[W_READ]] from %[[CONST_0]] : l9002, i32 -> l42
      // CHECK: moore.blocking_assign %v, %[[TMP2]] : l42
      %3 = moore.read %w : l9002
      moore.blocking_assign %2, %3 : l9002

      %4 = moore.constant 1 : i32
      %5 = moore.bool_cast %4 : i32 -> i1
      %6 = moore.conversion %5 : !moore.i1 -> i1
      scf.if %6 {
        // CHECK-NOT: moore.concat_ref
        %7 = moore.concat_ref %u, %v : (!moore.ref<l8960>, !moore.ref<l42>) -> <l9002>
        // CHECK: %[[W_READ:.+]] = moore.read %w : l9002
        %8 = moore.read %w : l9002
        // CHECK: %[[CONST_42:.+]] = moore.constant 42 : i32
        // CHECK: %[[TMP1:.+]] = moore.extract %[[W_READ]] from %[[CONST_42]] : l9002, i32 -> l8960
        // CHECK: moore.nonblocking_assign %u, %[[TMP1]] : l8960
        // CHECK: %[[CONST_0:.+]] = moore.constant 0 : i32
        // CHECK: %[[TMP2:.+]] = moore.extract %[[W_READ]] from %[[CONST_0]] : l9002, i32 -> l42
        // CHECK: moore.nonblocking_assign %v, %[[TMP2]] : l42
        moore.nonblocking_assign %7, %8 : l9002
      }
    }
    moore.output
  }
