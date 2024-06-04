// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @affine_store(
// CHECK-SAME:                                 %[[VAL_0:.*]]: index,
// CHECK-SAME:                                 %[[VAL_1:.*]]: none, ...) -> none attributes {argNames = ["in0", "inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_2:.*]] = memory[ld = 0, st = 1] (%[[VAL_3:.*]], %[[VAL_4:.*]]) {id = 0 : i32, lsq = false} : memref<10xf32>, (f32, index) -> none
// CHECK:           %[[VAL_5:.*]] = merge %[[VAL_0]] : index
// CHECK:           %[[VAL_6:.*]]:5 = fork [5] %[[VAL_1]] : none
// CHECK:           %[[VAL_7:.*]] = constant %[[VAL_6]]#3 {value = 1.100000e+01 : f32} : f32
// CHECK:           %[[VAL_8:.*]] = constant %[[VAL_6]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_9:.*]] = constant %[[VAL_6]]#1 {value = 10 : index} : index
// CHECK:           %[[VAL_10:.*]] = constant %[[VAL_6]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_11:.*]] = br %[[VAL_5]] : index
// CHECK:           %[[VAL_12:.*]] = br %[[VAL_6]]#4 : none
// CHECK:           %[[VAL_13:.*]] = br %[[VAL_7]] : f32
// CHECK:           %[[VAL_14:.*]] = br %[[VAL_8]] : index
// CHECK:           %[[VAL_15:.*]] = br %[[VAL_9]] : index
// CHECK:           %[[VAL_16:.*]] = br %[[VAL_10]] : index
// CHECK:           %[[VAL_17:.*]], %[[VAL_18:.*]] = control_merge %[[VAL_12]] : none
// CHECK:           %[[VAL_19:.*]]:5 = fork [5] %[[VAL_18]] : index
// CHECK:           %[[VAL_20:.*]] = buffer [1] seq %[[VAL_21:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_22:.*]]:6 = fork [6] %[[VAL_20]] : i1
// CHECK:           %[[VAL_23:.*]] = mux %[[VAL_22]]#5 {{\[}}%[[VAL_17]], %[[VAL_24:.*]]] : i1, none
// CHECK:           %[[VAL_25:.*]] = mux %[[VAL_19]]#4 {{\[}}%[[VAL_15]]] : index, index
// CHECK:           %[[VAL_26:.*]] = mux %[[VAL_22]]#4 {{\[}}%[[VAL_25]], %[[VAL_27:.*]]] : i1, index
// CHECK:           %[[VAL_28:.*]]:2 = fork [2] %[[VAL_26]] : index
// CHECK:           %[[VAL_29:.*]] = mux %[[VAL_19]]#3 {{\[}}%[[VAL_11]]] : index, index
// CHECK:           %[[VAL_30:.*]] = mux %[[VAL_22]]#3 {{\[}}%[[VAL_29]], %[[VAL_31:.*]]] : i1, index
// CHECK:           %[[VAL_32:.*]] = mux %[[VAL_19]]#2 {{\[}}%[[VAL_13]]] : index, f32
// CHECK:           %[[VAL_33:.*]] = mux %[[VAL_22]]#2 {{\[}}%[[VAL_32]], %[[VAL_34:.*]]] : i1, f32
// CHECK:           %[[VAL_35:.*]] = mux %[[VAL_19]]#1 {{\[}}%[[VAL_16]]] : index, index
// CHECK:           %[[VAL_36:.*]] = mux %[[VAL_22]]#1 {{\[}}%[[VAL_35]], %[[VAL_37:.*]]] : i1, index
// CHECK:           %[[VAL_38:.*]] = mux %[[VAL_19]]#0 {{\[}}%[[VAL_14]]] : index, index
// CHECK:           %[[VAL_39:.*]] = mux %[[VAL_22]]#0 {{\[}}%[[VAL_38]], %[[VAL_40:.*]]] : i1, index
// CHECK:           %[[VAL_41:.*]]:2 = fork [2] %[[VAL_39]] : index
// CHECK:           %[[VAL_21]] = merge %[[VAL_42:.*]]#0 : i1
// CHECK:           %[[VAL_43:.*]] = arith.cmpi slt, %[[VAL_41]]#0, %[[VAL_28]]#0 : index
// CHECK:           %[[VAL_42]]:7 = fork [7] %[[VAL_43]] : i1
// CHECK:           %[[VAL_44:.*]], %[[VAL_45:.*]] = cond_br %[[VAL_42]]#6, %[[VAL_28]]#1 : index
// CHECK:           sink %[[VAL_45]] : index
// CHECK:           %[[VAL_46:.*]], %[[VAL_47:.*]] = cond_br %[[VAL_42]]#5, %[[VAL_30]] : index
// CHECK:           sink %[[VAL_47]] : index
// CHECK:           %[[VAL_48:.*]], %[[VAL_49:.*]] = cond_br %[[VAL_42]]#4, %[[VAL_33]] : f32
// CHECK:           sink %[[VAL_49]] : f32
// CHECK:           %[[VAL_50:.*]], %[[VAL_51:.*]] = cond_br %[[VAL_42]]#3, %[[VAL_36]] : index
// CHECK:           sink %[[VAL_51]] : index
// CHECK:           %[[VAL_52:.*]], %[[VAL_53:.*]] = cond_br %[[VAL_42]]#2, %[[VAL_23]] : none
// CHECK:           %[[VAL_54:.*]], %[[VAL_55:.*]] = cond_br %[[VAL_42]]#1, %[[VAL_41]]#1 : index
// CHECK:           sink %[[VAL_55]] : index
// CHECK:           %[[VAL_56:.*]] = merge %[[VAL_46]] : index
// CHECK:           %[[VAL_57:.*]]:2 = fork [2] %[[VAL_56]] : index
// CHECK:           %[[VAL_58:.*]] = merge %[[VAL_54]] : index
// CHECK:           %[[VAL_59:.*]]:2 = fork [2] %[[VAL_58]] : index
// CHECK:           %[[VAL_60:.*]] = merge %[[VAL_48]] : f32
// CHECK:           %[[VAL_61:.*]]:2 = fork [2] %[[VAL_60]] : f32
// CHECK:           %[[VAL_62:.*]] = merge %[[VAL_50]] : index
// CHECK:           %[[VAL_63:.*]]:2 = fork [2] %[[VAL_62]] : index
// CHECK:           %[[VAL_64:.*]] = merge %[[VAL_44]] : index
// CHECK:           %[[VAL_65:.*]], %[[VAL_66:.*]] = control_merge %[[VAL_52]] : none
// CHECK:           %[[VAL_67:.*]]:2 = fork [2] %[[VAL_65]] : none
// CHECK:           %[[VAL_68:.*]]:3 = fork [3] %[[VAL_67]]#1 : none
// CHECK:           %[[VAL_69:.*]] = join %[[VAL_68]]#2, %[[VAL_2]] : none
// CHECK:           sink %[[VAL_66]] : index
// CHECK:           %[[VAL_70:.*]] = constant %[[VAL_68]]#1 {value = -1 : index} : index
// CHECK:           %[[VAL_71:.*]] = arith.muli %[[VAL_57]]#1, %[[VAL_70]] : index
// CHECK:           %[[VAL_72:.*]] = arith.addi %[[VAL_59]]#1, %[[VAL_71]] : index
// CHECK:           %[[VAL_73:.*]] = constant %[[VAL_68]]#0 {value = 7 : index} : index
// CHECK:           %[[VAL_74:.*]] = arith.addi %[[VAL_72]], %[[VAL_73]] : index
// CHECK:           %[[VAL_3]], %[[VAL_4]] = store {{\[}}%[[VAL_74]]] %[[VAL_61]]#1, %[[VAL_67]]#0 : index, f32
// CHECK:           %[[VAL_75:.*]] = arith.addi %[[VAL_59]]#0, %[[VAL_63]]#1 : index
// CHECK:           %[[VAL_31]] = br %[[VAL_57]]#0 : index
// CHECK:           %[[VAL_34]] = br %[[VAL_61]]#0 : f32
// CHECK:           %[[VAL_37]] = br %[[VAL_63]]#0 : index
// CHECK:           %[[VAL_27]] = br %[[VAL_64]] : index
// CHECK:           %[[VAL_24]] = br %[[VAL_69]] : none
// CHECK:           %[[VAL_40]] = br %[[VAL_75]] : index
// CHECK:           %[[VAL_76:.*]], %[[VAL_77:.*]] = control_merge %[[VAL_53]] : none
// CHECK:           sink %[[VAL_77]] : index
// CHECK:           return %[[VAL_76]] : none
// CHECK:         }
func.func @affine_store(%arg0: index) {
  %0 = memref.alloc() : memref<10xf32>
  %cst = arith.constant 1.100000e+01 : f32
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  cf.br ^bb1(%c0 : index)
^bb1(%1: index):      // 2 preds: ^bb0, ^bb2
  %2 = arith.cmpi slt, %1, %c10 : index
  cf.cond_br %2, ^bb2, ^bb3
^bb2: // pred: ^bb1
  %c-1 = arith.constant -1 : index
  %3 = arith.muli %arg0, %c-1 : index
  %4 = arith.addi %1, %3 : index
  %c7 = arith.constant 7 : index
  %5 = arith.addi %4, %c7 : index
  memref.store %cst, %0[%5] : memref<10xf32>
  %6 = arith.addi %1, %c1 : index
  cf.br ^bb1(%6 : index)
^bb3: // pred: ^bb1
  return
}