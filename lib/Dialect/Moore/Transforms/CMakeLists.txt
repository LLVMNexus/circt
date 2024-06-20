add_circt_dialect_library(CIRCTMooreTransforms
SimplifyProcedures.cpp
SimplifyAssigns.cpp


  DEPENDS
  CIRCTMooreTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTMoore
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
