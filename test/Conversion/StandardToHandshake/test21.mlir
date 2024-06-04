// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @loop_min_max(
// CHECK-SAME:                                 %[[VAL_0:.*]]: index,
// CHECK-SAME:                                 %[[VAL_1:.*]]: none, ...) -> none attributes {argNames = ["in0", "inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : index
// CHECK:           %[[VAL_3:.*]]:4 = fork [4] %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_3]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_5:.*]] = constant %[[VAL_3]]#1 {value = 42 : index} : index
// CHECK:           %[[VAL_6:.*]] = constant %[[VAL_3]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_7:.*]] = br %[[VAL_2]] : index
// CHECK:           %[[VAL_8:.*]] = br %[[VAL_3]]#3 : none
// CHECK:           %[[VAL_9:.*]] = br %[[VAL_4]] : index
// CHECK:           %[[VAL_10:.*]] = br %[[VAL_5]] : index
// CHECK:           %[[VAL_11:.*]] = br %[[VAL_6]] : index
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = control_merge %[[VAL_8]] : none
// CHECK:           %[[VAL_14:.*]]:4 = fork [4] %[[VAL_13]] : index
// CHECK:           %[[VAL_15:.*]] = buffer [1] seq %[[VAL_16:.*]] {initValues = [0]} : i1
// CHECK:           %[[VAL_17:.*]]:5 = fork [5] %[[VAL_15]] : i1
// CHECK:           %[[VAL_18:.*]] = mux %[[VAL_17]]#4 {{\[}}%[[VAL_12]], %[[VAL_19:.*]]] : i1, none
// CHECK:           %[[VAL_20:.*]] = mux %[[VAL_14]]#3 {{\[}}%[[VAL_10]]] : index, index
// CHECK:           %[[VAL_21:.*]] = mux %[[VAL_17]]#3 {{\[}}%[[VAL_20]], %[[VAL_22:.*]]] : i1, index
// CHECK:           %[[VAL_23:.*]]:2 = fork [2] %[[VAL_21]] : index
// CHECK:           %[[VAL_24:.*]] = mux %[[VAL_14]]#2 {{\[}}%[[VAL_7]]] : index, index
// CHECK:           %[[VAL_25:.*]] = mux %[[VAL_17]]#2 {{\[}}%[[VAL_24]], %[[VAL_26:.*]]] : i1, index
// CHECK:           %[[VAL_27:.*]] = mux %[[VAL_14]]#1 {{\[}}%[[VAL_11]]] : index, index
// CHECK:           %[[VAL_28:.*]] = mux %[[VAL_17]]#1 {{\[}}%[[VAL_27]], %[[VAL_29:.*]]] : i1, index
// CHECK:           %[[VAL_30:.*]] = mux %[[VAL_14]]#0 {{\[}}%[[VAL_9]]] : index, index
// CHECK:           %[[VAL_31:.*]] = mux %[[VAL_17]]#0 {{\[}}%[[VAL_30]], %[[VAL_32:.*]]] : i1, index
// CHECK:           %[[VAL_33:.*]]:2 = fork [2] %[[VAL_31]] : index
// CHECK:           %[[VAL_16]] = merge %[[VAL_34:.*]]#0 : i1
// CHECK:           %[[VAL_35:.*]] = arith.cmpi slt, %[[VAL_33]]#0, %[[VAL_23]]#0 : index
// CHECK:           %[[VAL_34]]:6 = fork [6] %[[VAL_35]] : i1
// CHECK:           %[[VAL_36:.*]], %[[VAL_37:.*]] = cond_br %[[VAL_34]]#5, %[[VAL_23]]#1 : index
// CHECK:           sink %[[VAL_37]] : index
// CHECK:           %[[VAL_38:.*]], %[[VAL_39:.*]] = cond_br %[[VAL_34]]#4, %[[VAL_25]] : index
// CHECK:           sink %[[VAL_39]] : index
// CHECK:           %[[VAL_40:.*]], %[[VAL_41:.*]] = cond_br %[[VAL_34]]#3, %[[VAL_28]] : index
// CHECK:           sink %[[VAL_41]] : index
// CHECK:           %[[VAL_42:.*]], %[[VAL_43:.*]] = cond_br %[[VAL_34]]#2, %[[VAL_18]] : none
// CHECK:           %[[VAL_44:.*]], %[[VAL_45:.*]] = cond_br %[[VAL_34]]#1, %[[VAL_33]]#1 : index
// CHECK:           sink %[[VAL_45]] : index
// CHECK:           %[[VAL_46:.*]] = merge %[[VAL_44]] : index
// CHECK:           %[[VAL_47:.*]]:5 = fork [5] %[[VAL_46]] : index
// CHECK:           %[[VAL_48:.*]] = merge %[[VAL_38]] : index
// CHECK:           %[[VAL_49:.*]]:4 = fork [4] %[[VAL_48]] : index
// CHECK:           %[[VAL_50:.*]] = merge %[[VAL_40]] : index
// CHECK:           %[[VAL_51:.*]] = merge %[[VAL_36]] : index
// CHECK:           %[[VAL_52:.*]], %[[VAL_53:.*]] = control_merge %[[VAL_42]] : none
// CHECK:           %[[VAL_54:.*]]:4 = fork [4] %[[VAL_52]] : none
// CHECK:           sink %[[VAL_53]] : index
// CHECK:           %[[VAL_55:.*]] = constant %[[VAL_54]]#2 {value = -1 : index} : index
// CHECK:           %[[VAL_56:.*]] = arith.muli %[[VAL_47]]#4, %[[VAL_55]] : index
// CHECK:           %[[VAL_57:.*]] = arith.addi %[[VAL_56]], %[[VAL_49]]#3 : index
// CHECK:           %[[VAL_58:.*]]:2 = fork [2] %[[VAL_57]] : index
// CHECK:           %[[VAL_59:.*]] = arith.cmpi sgt, %[[VAL_47]]#3, %[[VAL_58]]#1 : index
// CHECK:           %[[VAL_60:.*]] = select %[[VAL_59]], %[[VAL_58]]#0, %[[VAL_47]]#2 : index
// CHECK:           %[[VAL_61:.*]] = constant %[[VAL_54]]#1 {value = 10 : index} : index
// CHECK:           %[[VAL_62:.*]] = arith.addi %[[VAL_47]]#1, %[[VAL_61]] : index
// CHECK:           %[[VAL_63:.*]]:2 = fork [2] %[[VAL_62]] : index
// CHECK:           %[[VAL_64:.*]] = arith.cmpi slt, %[[VAL_49]]#2, %[[VAL_63]]#1 : index
// CHECK:           %[[VAL_65:.*]] = select %[[VAL_64]], %[[VAL_63]]#0, %[[VAL_49]]#1 : index
// CHECK:           %[[VAL_66:.*]] = constant %[[VAL_54]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_67:.*]] = br %[[VAL_47]]#0 : index
// CHECK:           %[[VAL_68:.*]] = br %[[VAL_49]]#0 : index
// CHECK:           %[[VAL_69:.*]] = br %[[VAL_50]] : index
// CHECK:           %[[VAL_70:.*]] = br %[[VAL_51]] : index
// CHECK:           %[[VAL_71:.*]] = br %[[VAL_54]]#3 : none
// CHECK:           %[[VAL_72:.*]] = br %[[VAL_60]] : index
// CHECK:           %[[VAL_73:.*]] = br %[[VAL_65]] : index
// CHECK:           %[[VAL_74:.*]] = br %[[VAL_66]] : index
// CHECK:           %[[VAL_75:.*]] = mux %[[VAL_76:.*]]#6 {{\[}}%[[VAL_77:.*]], %[[VAL_73]]] : index, index
// CHECK:           %[[VAL_78:.*]]:2 = fork [2] %[[VAL_75]] : index
// CHECK:           %[[VAL_79:.*]] = mux %[[VAL_76]]#5 {{\[}}%[[VAL_80:.*]], %[[VAL_74]]] : index, index
// CHECK:           %[[VAL_81:.*]] = mux %[[VAL_76]]#4 {{\[}}%[[VAL_82:.*]], %[[VAL_67]]] : index, index
// CHECK:           %[[VAL_83:.*]] = mux %[[VAL_76]]#3 {{\[}}%[[VAL_84:.*]], %[[VAL_69]]] : index, index
// CHECK:           %[[VAL_85:.*]] = mux %[[VAL_76]]#2 {{\[}}%[[VAL_86:.*]], %[[VAL_70]]] : index, index
// CHECK:           %[[VAL_87:.*]] = mux %[[VAL_76]]#1 {{\[}}%[[VAL_88:.*]], %[[VAL_68]]] : index, index
// CHECK:           %[[VAL_89:.*]], %[[VAL_90:.*]] = control_merge %[[VAL_91:.*]], %[[VAL_71]] : none
// CHECK:           %[[VAL_76]]:7 = fork [7] %[[VAL_90]] : index
// CHECK:           %[[VAL_92:.*]] = mux %[[VAL_76]]#0 {{\[}}%[[VAL_93:.*]], %[[VAL_72]]] : index, index
// CHECK:           %[[VAL_94:.*]]:2 = fork [2] %[[VAL_92]] : index
// CHECK:           %[[VAL_95:.*]] = arith.cmpi slt, %[[VAL_94]]#1, %[[VAL_78]]#1 : index
// CHECK:           %[[VAL_96:.*]]:8 = fork [8] %[[VAL_95]] : i1
// CHECK:           %[[VAL_97:.*]], %[[VAL_98:.*]] = cond_br %[[VAL_96]]#7, %[[VAL_78]]#0 : index
// CHECK:           sink %[[VAL_98]] : index
// CHECK:           %[[VAL_99:.*]], %[[VAL_100:.*]] = cond_br %[[VAL_96]]#6, %[[VAL_79]] : index
// CHECK:           sink %[[VAL_100]] : index
// CHECK:           %[[VAL_101:.*]], %[[VAL_102:.*]] = cond_br %[[VAL_96]]#5, %[[VAL_81]] : index
// CHECK:           %[[VAL_103:.*]], %[[VAL_104:.*]] = cond_br %[[VAL_96]]#4, %[[VAL_83]] : index
// CHECK:           %[[VAL_105:.*]], %[[VAL_106:.*]] = cond_br %[[VAL_96]]#3, %[[VAL_85]] : index
// CHECK:           %[[VAL_107:.*]], %[[VAL_108:.*]] = cond_br %[[VAL_96]]#2, %[[VAL_87]] : index
// CHECK:           %[[VAL_109:.*]], %[[VAL_110:.*]] = cond_br %[[VAL_96]]#1, %[[VAL_89]] : none
// CHECK:           %[[VAL_111:.*]], %[[VAL_112:.*]] = cond_br %[[VAL_96]]#0, %[[VAL_94]]#0 : index
// CHECK:           sink %[[VAL_112]] : index
// CHECK:           %[[VAL_113:.*]] = merge %[[VAL_111]] : index
// CHECK:           %[[VAL_114:.*]] = merge %[[VAL_99]] : index
// CHECK:           %[[VAL_115:.*]]:2 = fork [2] %[[VAL_114]] : index
// CHECK:           %[[VAL_116:.*]] = merge %[[VAL_97]] : index
// CHECK:           %[[VAL_117:.*]] = merge %[[VAL_101]] : index
// CHECK:           %[[VAL_118:.*]] = merge %[[VAL_103]] : index
// CHECK:           %[[VAL_119:.*]] = merge %[[VAL_105]] : index
// CHECK:           %[[VAL_120:.*]] = merge %[[VAL_107]] : index
// CHECK:           %[[VAL_121:.*]], %[[VAL_122:.*]] = control_merge %[[VAL_109]] : none
// CHECK:           sink %[[VAL_122]] : index
// CHECK:           %[[VAL_123:.*]] = arith.addi %[[VAL_113]], %[[VAL_115]]#1 : index
// CHECK:           %[[VAL_80]] = br %[[VAL_115]]#0 : index
// CHECK:           %[[VAL_77]] = br %[[VAL_116]] : index
// CHECK:           %[[VAL_82]] = br %[[VAL_117]] : index
// CHECK:           %[[VAL_84]] = br %[[VAL_118]] : index
// CHECK:           %[[VAL_86]] = br %[[VAL_119]] : index
// CHECK:           %[[VAL_88]] = br %[[VAL_120]] : index
// CHECK:           %[[VAL_91]] = br %[[VAL_121]] : none
// CHECK:           %[[VAL_93]] = br %[[VAL_123]] : index
// CHECK:           %[[VAL_124:.*]] = merge %[[VAL_102]] : index
// CHECK:           %[[VAL_125:.*]] = merge %[[VAL_104]] : index
// CHECK:           %[[VAL_126:.*]]:2 = fork [2] %[[VAL_125]] : index
// CHECK:           %[[VAL_127:.*]] = merge %[[VAL_106]] : index
// CHECK:           %[[VAL_128:.*]] = merge %[[VAL_108]] : index
// CHECK:           %[[VAL_129:.*]], %[[VAL_130:.*]] = control_merge %[[VAL_110]] : none
// CHECK:           sink %[[VAL_130]] : index
// CHECK:           %[[VAL_131:.*]] = arith.addi %[[VAL_124]], %[[VAL_126]]#1 : index
// CHECK:           %[[VAL_29]] = br %[[VAL_126]]#0 : index
// CHECK:           %[[VAL_22]] = br %[[VAL_127]] : index
// CHECK:           %[[VAL_26]] = br %[[VAL_128]] : index
// CHECK:           %[[VAL_19]] = br %[[VAL_129]] : none
// CHECK:           %[[VAL_32]] = br %[[VAL_131]] : index
// CHECK:           %[[VAL_132:.*]], %[[VAL_133:.*]] = control_merge %[[VAL_43]] : none
// CHECK:           sink %[[VAL_133]] : index
// CHECK:           return %[[VAL_132]] : none
// CHECK:         }
func.func @loop_min_max(%arg0: index) {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : index
  %c1 = arith.constant 1 : index
  cf.br ^bb1(%c0 : index)
^bb1(%0: index):      // 2 preds: ^bb0, ^bb5
  %1 = arith.cmpi slt, %0, %c42 : index
  cf.cond_br %1, ^bb2, ^bb6
^bb2: // pred: ^bb1
  %c-1 = arith.constant -1 : index
  %2 = arith.muli %0, %c-1 : index
  %3 = arith.addi %2, %arg0 : index
  %4 = arith.cmpi sgt, %0, %3 : index
  %5 = arith.select %4, %0, %3 : index
  %c10 = arith.constant 10 : index
  %6 = arith.addi %0, %c10 : index
  %7 = arith.cmpi slt, %arg0, %6 : index
  %8 = arith.select %7, %arg0, %6 : index
  %c1_0 = arith.constant 1 : index
  cf.br ^bb3(%5 : index)
^bb3(%9: index):      // 2 preds: ^bb2, ^bb4
  %10 = arith.cmpi slt, %9, %8 : index
  cf.cond_br %10, ^bb4, ^bb5
^bb4: // pred: ^bb3
  %11 = arith.addi %9, %c1_0 : index
  cf.br ^bb3(%11 : index)
^bb5: // pred: ^bb3
  %12 = arith.addi %0, %c1 : index
  cf.br ^bb1(%12 : index)
^bb6: // pred: ^bb1
  return
}