set(COMMON ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
idf_component_register(
  SRCS "main.cpp" "${COMMON}/coda_runtime.cc" "${COMMON}/coda_score.c" "${COMMON}/model_data.cc"
  INCLUDE_DIRS "." "${COMMON}"
  PRIV_REQUIRES esp_driver_uart esp_hw_support)
target_compile_options(${COMPONENT_LIB} PRIVATE -ffp-contract=off -O2
  -DCODA_BACKEND="esp-nn" -DCODA_ARENA_BYTES=65536 -DCODA_MODEL_CAPACITY=131072)
