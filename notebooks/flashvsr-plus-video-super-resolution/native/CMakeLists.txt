cmake_minimum_required(VERSION 3.20)
project(flashvsr_plus_openvino_native LANGUAGES CXX)

# This is intentionally a small source-review build, not the OTC application
# build.  It never builds a video worker or runs a GPU inference from CTest.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FLASH_POST_BUILD_TESTS "Build the CPU structural and stream tests" ON)
option(FLASH_POST_BUILD_PREPARER
       "Build the optional BF16 converter and authenticated preparation library"
       OFF)

set(FLASH_ASSET_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../assets/flash-assets")
set(FLASH_SCRIPT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../scripts")
set(FLASH_ASSET_INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../assets")

find_package(OpenSSL REQUIRED COMPONENTS Crypto)

add_library(flashvsr-plus-openvino-assets STATIC
  seedvr_assets.cpp
  verified_kernel_files.cpp
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_sources(flashvsr-plus-openvino-assets PRIVATE flash_history.cpp)
endif()
target_include_directories(flashvsr-plus-openvino-assets PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${FLASH_ASSET_INCLUDE_ROOT}")
target_link_libraries(flashvsr-plus-openvino-assets PUBLIC OpenSSL::Crypto)
target_compile_options(flashvsr-plus-openvino-assets PRIVATE -Wall -Wextra -Wpedantic)

find_package(OpenVINO CONFIG REQUIRED)

add_library(flashvsr-plus-openvino STATIC
  flash_ops.cpp
  flash_model.cpp
  flash_stage.cpp
  flash_pre_model.cpp
  flash_pre_session.cpp
  flash_post_session.cpp
  flash_lq_model.cpp
  flash_lq_session.cpp
  flash_cross_model.cpp
  flash_cross_session.cpp
  flash_edges_model.cpp
  flash_edges_session.cpp
  flash_geometry.cpp
  flash_mask.cpp
  flash_rope.cpp
  flash_text_model.cpp
  flash_text_session.cpp
  flash_decoder_model.cpp
  flash_decoder_frame.cpp
  flash_decoder_session.cpp
  flash_source.cpp
  flash_tokens.cpp
  flash_token_session.cpp
  flash_attention_session.cpp
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Compile the controller API for source/lifetime coverage only.  This
  # reference build deliberately has no product main.cpp, worker executable,
  # or GPU launch target.
  target_sources(flashvsr-plus-openvino PRIVATE flash_denoiser.cpp flash_video.cpp)
endif()
target_include_directories(flashvsr-plus-openvino PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${FLASH_ASSET_INCLUDE_ROOT}")
target_link_libraries(flashvsr-plus-openvino PUBLIC flashvsr-plus-openvino-assets openvino::runtime)
target_compile_options(flashvsr-plus-openvino PRIVATE -Wall -Wextra -Wpedantic)

if(FLASH_POST_BUILD_PREPARER)
  # These sources are build-time only.  They authenticate external safetensors
  # and produce the generated table; no learned payload is bundled here.
  add_library(flashvsr-plus-openvino-preparation STATIC
    seedvr_prepare.cpp
    flash_prepare.cpp
    flash_prepare_cli.cpp
  )
  target_include_directories(flashvsr-plus-openvino-preparation PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${FLASH_ASSET_INCLUDE_ROOT}")
  target_link_libraries(flashvsr-plus-openvino-preparation PUBLIC flashvsr-plus-openvino-assets)
  target_compile_options(flashvsr-plus-openvino-preparation PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_definitions(flashvsr-plus-openvino-preparation PRIVATE OTC_FLASH_PREPARE_TESTING)

  add_executable(flashvsr-plus-openvino-bf16-converter EXCLUDE_FROM_ALL bf16_convert.cpp)
  target_include_directories(flashvsr-plus-openvino-bf16-converter PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(flashvsr-plus-openvino-bf16-converter PRIVATE flashvsr-plus-openvino-preparation)
  target_compile_options(flashvsr-plus-openvino-bf16-converter PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(flashvsr-plus-openvino-prepare-structural-tests seedvr_prepare_tests.cpp)
  target_link_libraries(flashvsr-plus-openvino-prepare-structural-tests PRIVATE flashvsr-plus-openvino-preparation)
  target_compile_options(flashvsr-plus-openvino-prepare-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  if(FLASH_POST_BUILD_TESTS)
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/run-prepare-structural-tests.cmake" CONTENT
"string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef fixture)
execute_process(COMMAND \"$<TARGET_FILE:flashvsr-plus-openvino-prepare-structural-tests>\" \"${CMAKE_CURRENT_BINARY_DIR}/prepare-structural-fixture-\${fixture}\" RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  message(FATAL_ERROR \"Preparation structural tests failed: \${result}\")
endif()
")
    add_test(NAME flashvsr-plus-openvino-prepare-structural
             COMMAND "${CMAKE_COMMAND}" -P
                     "${CMAKE_CURRENT_BINARY_DIR}/run-prepare-structural-tests.cmake")

    add_executable(flashvsr-plus-openvino-flash-prepare-structural-tests flash_prepare_tests.cpp)
    target_link_libraries(flashvsr-plus-openvino-flash-prepare-structural-tests PRIVATE
                          flashvsr-plus-openvino-preparation)
    target_compile_options(flashvsr-plus-openvino-flash-prepare-structural-tests PRIVATE
      -Wall -Wextra -Wpedantic -UNDEBUG)
    add_test(NAME flashvsr-plus-openvino-flash-prepare-structural
             COMMAND flashvsr-plus-openvino-flash-prepare-structural-tests)

    add_executable(flashvsr-plus-openvino-prepare-cli-structural-tests
                   flash_prepare_cli_tests.cpp)
    target_link_libraries(flashvsr-plus-openvino-prepare-cli-structural-tests PRIVATE
                          flashvsr-plus-openvino-preparation)
    target_compile_definitions(flashvsr-plus-openvino-prepare-cli-structural-tests PRIVATE
                               OTC_FLASH_PREPARE_TESTING)
    target_compile_options(flashvsr-plus-openvino-prepare-cli-structural-tests PRIVATE
      -Wall -Wextra -Wpedantic -UNDEBUG)
    add_test(NAME flashvsr-plus-openvino-prepare-cli-structural
             COMMAND flashvsr-plus-openvino-prepare-cli-structural-tests
                     "${FLASH_ASSET_ROOT}")
  endif()
endif()

if(FLASH_POST_BUILD_TESTS)
  enable_testing()

  # The decoder contract test is stdlib-only; keep it optional so the native
  # structural build remains usable without a Python interpreter.
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME flashvsr-plus-openvino-decoder-export
             COMMAND "${Python3_EXECUTABLE}"
                     "${FLASH_SCRIPT_ROOT}/test_export_flash_decoder.py")
    add_test(NAME flashvsr-plus-openvino-decoder-prepare
             COMMAND "${Python3_EXECUTABLE}"
                     "${FLASH_SCRIPT_ROOT}/test_prepare_flash_decoder.py")
    add_test(NAME flashvsr-plus-openvino-conditioning-export
             COMMAND "${Python3_EXECUTABLE}"
                     "${FLASH_SCRIPT_ROOT}/test_export_flash_conditioning.py")
  endif()

  add_executable(flashvsr-plus-openvino-stream-tests flash_stream_tests.cpp)
  target_compile_options(flashvsr-plus-openvino-stream-tests PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-stream COMMAND flashvsr-plus-openvino-stream-tests)

  add_executable(flashvsr-plus-openvino-source-tests flash_source_test.cpp flash_source.cpp)
  target_compile_options(flashvsr-plus-openvino-source-tests PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-source COMMAND flashvsr-plus-openvino-source-tests)

  add_executable(flashvsr-plus-openvino-rope-structural-tests flash_rope_tests.cpp
                 flash_rope.cpp flash_geometry.cpp)
  target_compile_options(flashvsr-plus-openvino-rope-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-rope-structural COMMAND flashvsr-plus-openvino-rope-structural-tests)

  add_executable(flashvsr-plus-openvino-ops-structural-tests flash_ops_tests.cpp flash_ops.cpp)
  target_link_libraries(flashvsr-plus-openvino-ops-structural-tests PRIVATE openvino::runtime)
  target_compile_options(flashvsr-plus-openvino-ops-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-ops-structural
           COMMAND flashvsr-plus-openvino-ops-structural-tests)

  add_executable(flashvsr-plus-openvino-session-structural-tests flash_post_session_tests.cpp)
  target_link_libraries(flashvsr-plus-openvino-session-structural-tests PRIVATE flashvsr-plus-openvino)
  target_compile_options(flashvsr-plus-openvino-session-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-session-structural
           COMMAND flashvsr-plus-openvino-session-structural-tests
                   "${FLASH_ASSET_ROOT}")

  add_executable(flashvsr-plus-openvino-pre-model-structural-tests flash_pre_model_tests.cpp)
  target_link_libraries(flashvsr-plus-openvino-pre-model-structural-tests PRIVATE flashvsr-plus-openvino)
  target_compile_options(flashvsr-plus-openvino-pre-model-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-pre-model-structural
           COMMAND flashvsr-plus-openvino-pre-model-structural-tests
                   "${FLASH_ASSET_ROOT}")

  add_executable(flashvsr-plus-openvino-model-finite-tests flash_model_tests.cpp)
  target_link_libraries(flashvsr-plus-openvino-model-finite-tests PRIVATE flashvsr-plus-openvino)
  target_compile_options(flashvsr-plus-openvino-model-finite-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-model-finite COMMAND flashvsr-plus-openvino-model-finite-tests)

  add_executable(flashvsr-plus-openvino-lq-model-structural-tests flash_lq_model_tests.cpp)
  target_link_libraries(flashvsr-plus-openvino-lq-model-structural-tests PRIVATE flashvsr-plus-openvino)
  target_compile_options(flashvsr-plus-openvino-lq-model-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-lq-model-structural
           COMMAND flashvsr-plus-openvino-lq-model-structural-tests
                   "${FLASH_ASSET_ROOT}")

  add_executable(flashvsr-plus-openvino-cross-model-structural-tests flash_cross_model_tests.cpp)
  target_link_libraries(flashvsr-plus-openvino-cross-model-structural-tests PRIVATE flashvsr-plus-openvino)
  target_compile_options(flashvsr-plus-openvino-cross-model-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-cross-model-structural
           COMMAND flashvsr-plus-openvino-cross-model-structural-tests
                   "${FLASH_ASSET_ROOT}")

  add_executable(flashvsr-plus-openvino-edges-model-structural-tests flash_edges_model_tests.cpp)
  target_link_libraries(flashvsr-plus-openvino-edges-model-structural-tests PRIVATE flashvsr-plus-openvino)
  target_compile_options(flashvsr-plus-openvino-edges-model-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-edges-model-structural
           COMMAND flashvsr-plus-openvino-edges-model-structural-tests
                   "${FLASH_ASSET_ROOT}")

  add_executable(flashvsr-plus-openvino-geometry-structural-tests flash_geometry_tests.cpp
                 flash_geometry.cpp)
  target_include_directories(flashvsr-plus-openvino-geometry-structural-tests PRIVATE
                             "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_options(flashvsr-plus-openvino-geometry-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-geometry-structural
           COMMAND flashvsr-plus-openvino-geometry-structural-tests)

  add_executable(flashvsr-plus-openvino-mask-structural-tests flash_mask_tests.cpp flash_mask.cpp)
  target_compile_options(flashvsr-plus-openvino-mask-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-mask-structural COMMAND flashvsr-plus-openvino-mask-structural-tests)

  add_executable(flashvsr-plus-openvino-token-structural-tests flash_tokens_tests.cpp flash_tokens.cpp)
  target_compile_options(flashvsr-plus-openvino-token-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-token-structural COMMAND flashvsr-plus-openvino-token-structural-tests)

  add_executable(flashvsr-plus-openvino-decoder-model-structural-tests flash_decoder_model_tests.cpp)
  target_link_libraries(flashvsr-plus-openvino-decoder-model-structural-tests PRIVATE flashvsr-plus-openvino)
  target_compile_options(flashvsr-plus-openvino-decoder-model-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-decoder-model-structural
           COMMAND flashvsr-plus-openvino-decoder-model-structural-tests
                   "${FLASH_ASSET_ROOT}")

  add_executable(flashvsr-plus-openvino-decoder-frame-structural-tests
                 flash_decoder_frame_tests.cpp flash_decoder_frame.cpp
                 flash_geometry.cpp)
  target_compile_options(flashvsr-plus-openvino-decoder-frame-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-decoder-frame-structural
           COMMAND flashvsr-plus-openvino-decoder-frame-structural-tests)

  add_executable(flashvsr-plus-openvino-stage-structural-tests flash_stage_tests.cpp)
  target_link_libraries(flashvsr-plus-openvino-stage-structural-tests PRIVATE flashvsr-plus-openvino)
  target_compile_options(flashvsr-plus-openvino-stage-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-stage-structural COMMAND flashvsr-plus-openvino-stage-structural-tests)

  add_executable(flashvsr-plus-openvino-text-model-structural-tests flash_text_model_tests.cpp)
  target_link_libraries(flashvsr-plus-openvino-text-model-structural-tests PRIVATE flashvsr-plus-openvino)
  target_compile_options(flashvsr-plus-openvino-text-model-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-text-model-structural
           COMMAND flashvsr-plus-openvino-text-model-structural-tests
                   "${FLASH_ASSET_ROOT}")

  add_executable(flashvsr-plus-openvino-setup-structural-tests flash_setup_tests.cpp)
  target_link_libraries(flashvsr-plus-openvino-setup-structural-tests PRIVATE flashvsr-plus-openvino)
  target_compile_options(flashvsr-plus-openvino-setup-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME flashvsr-plus-openvino-setup-structural COMMAND flashvsr-plus-openvino-setup-structural-tests)

  add_executable(flashvsr-plus-openvino-kernel-structural-tests verified_kernel_files_tests.cpp)
  target_link_libraries(flashvsr-plus-openvino-kernel-structural-tests PRIVATE flashvsr-plus-openvino-assets)
  target_compile_options(flashvsr-plus-openvino-kernel-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  # The test creates and removes a fresh fixture directory itself.  Passing a
  # build-tree leaf keeps it independent of the caller's working directory.
  add_test(NAME flashvsr-plus-openvino-kernel-structural
           COMMAND flashvsr-plus-openvino-kernel-structural-tests
                   "${CMAKE_CURRENT_BINARY_DIR}/kernel-structural-fixture")

  add_executable(flashvsr-plus-openvino-assets-structural-tests seedvr_assets_tests.cpp)
  target_link_libraries(flashvsr-plus-openvino-assets-structural-tests PRIVATE flashvsr-plus-openvino-assets)
  target_compile_options(flashvsr-plus-openvino-assets-structural-tests PRIVATE
    -Wall -Wextra -Wpedantic -UNDEBUG)
  file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/run-assets-structural-tests.cmake" CONTENT
"string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef fixture)
execute_process(COMMAND \"$<TARGET_FILE:flashvsr-plus-openvino-assets-structural-tests>\" \"${CMAKE_CURRENT_BINARY_DIR}/asset-structural-fixture-\${fixture}\" RESULT_VARIABLE result)
if(NOT result EQUAL 0)
  message(FATAL_ERROR \"Asset structural tests failed: \${result}\")
endif()
")
  add_test(NAME flashvsr-plus-openvino-assets-structural
           COMMAND "${CMAKE_COMMAND}" -P
                   "${CMAKE_CURRENT_BINARY_DIR}/run-assets-structural-tests.cmake")

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(flashvsr-plus-openvino-history-structural-tests flash_history_tests.cpp)
    target_link_libraries(flashvsr-plus-openvino-history-structural-tests PRIVATE flashvsr-plus-openvino-assets)
    target_compile_options(flashvsr-plus-openvino-history-structural-tests PRIVATE
      -Wall -Wextra -Wpedantic -UNDEBUG)
    add_test(NAME flashvsr-plus-openvino-history-structural
             COMMAND flashvsr-plus-openvino-history-structural-tests
                     "${CMAKE_CURRENT_BINARY_DIR}")
  endif()
endif()
