add_circt_dialect_library(CIRCTSimTransforms
  SeparateImportAndCall.cpp

  DEPENDS
  CIRCTSimTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTSupport
  CIRCTSV
  CIRCTSim
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
