add_circt_conversion_library(CIRCTMooreToCore
  MooreToCore.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTMoore
  CIRCTLLHD
  CIRCTHW
  CIRCTComb
  CIRCTSeq
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRTransforms
)
