cmake_minimum_required(VERSION 3.10)

project(llama.rn)

set(CMAKE_CXX_STANDARD 11)
set(RNLLAMA_LIB_DIR ${CMAKE_SOURCE_DIR}/../../../cpp)

include_directories(${RNLLAMA_LIB_DIR})

set(
    SOURCE_FILES
    ${RNLLAMA_LIB_DIR}/ggml-alloc.c
    ${RNLLAMA_LIB_DIR}/ggml-backend.c
    ${RNLLAMA_LIB_DIR}/ggml.c

    ${RNLLAMA_LIB_DIR}/ggml-opencl.cpp
    ${RNLLAMA_LIB_DIR}/k_quants.c

    ${RNLLAMA_LIB_DIR}/common.cpp
    ${RNLLAMA_LIB_DIR}/grammar-parser.cpp
    ${RNLLAMA_LIB_DIR}/sampling.cpp
    ${RNLLAMA_LIB_DIR}/llama.cpp
    ${RNLLAMA_LIB_DIR}/rn-llama.hpp
    ${CMAKE_SOURCE_DIR}/jni.cpp
)


set(KERNELS copy_fast copy_pad transpose_fast transpose_pad xaxpy xdot xger
            xgemm xgemm_direct xgemv invert xconvgemm)
set(DATABASES copy pad padtranspose transpose xaxpy xdot
              xgemm xgemm_direct xgemv xgemv_fast xgemv_fast_rot xger invert
              gemm_routine trsv_routine xconvgemm)
set(LEVEL1_ROUTINES xswap xscal xcopy xaxpy xdot xdotu xdotc xnrm2 xasum xamax)
set(LEVEL2_ROUTINES xgemv xgbmv xhemv xhbmv xhpmv xsymv xsbmv xspmv xtrmv xtbmv xtpmv xtrsv
                    xger xgeru xgerc xher xhpr xher2 xhpr2 xsyr xspr xsyr2 xspr2)
set(LEVEL3_ROUTINES xgemm xsymm xhemm xsyrk xherk xsyr2k xher2k xtrmm xtrsm)
set(LEVELX_ROUTINES xhad xomatcopy xim2col xcol2im xconvgemm xaxpybatched xgemmbatched xgemmstridedbatched)
set(ROUTINES ${LEVEL1_ROUTINES} ${LEVEL2_ROUTINES} ${LEVEL3_ROUTINES} ${LEVELX_ROUTINES})

set(OPENCL_SOURCES
  ${RNLLAMA_LIB_DIR}/CLBlast/database/database.cpp
  ${RNLLAMA_LIB_DIR}/CLBlast/routines/common.cpp
  ${RNLLAMA_LIB_DIR}/CLBlast/utilities/compile.cpp
  ${RNLLAMA_LIB_DIR}/CLBlast/utilities/clblast_exceptions.cpp
  ${RNLLAMA_LIB_DIR}/CLBlast/utilities/timing.cpp
  ${RNLLAMA_LIB_DIR}/CLBlast/utilities/utilities.cpp
#   ${RNLLAMA_LIB_DIR}/CLBlast/api_common.cpp
  ${RNLLAMA_LIB_DIR}/CLBlast/cache.cpp
  ${RNLLAMA_LIB_DIR}/CLBlast/kernel_preprocessor.cpp
  ${RNLLAMA_LIB_DIR}/CLBlast/routine.cpp
  ${RNLLAMA_LIB_DIR}/CLBlast/routines/levelx/xinvert.cpp  # only source, don't include it as a test
  ${RNLLAMA_LIB_DIR}/CLBlast/tuning/configurations.cpp
)
set(OPENCL_HEADERS  # such that they can be discovered by IDEs such as CLion and Visual Studio
    ${RNLLAMA_LIB_DIR}/CLBlast/clblast_half.h

    ${RNLLAMA_LIB_DIR}/CLBlast/database/apple_cpu_fallback.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/database/database.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/database/database_structure.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/routines/level1/xamin.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/routines/level1/xmax.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/routines/level1/xmin.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/routines/level1/xsum.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/routines/common.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/routines/routines.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/utilities/buffer_test.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/utilities/compile.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/utilities/clblast_exceptions.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/utilities/device_mapping.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/utilities/msvc.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/utilities/timing.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/utilities/utilities.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/cache.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/kernel_preprocessor.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/cxpp11_common.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/routine.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/tuning/configurations.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/tuning/tuning.hpp
    ${RNLLAMA_LIB_DIR}/CLBlast/tuning/routines/routine_tuner.hpp
)

set(OPENCL_SOURCES ${OPENCL_SOURCES} ${RNLLAMA_LIB_DIR}/CLBlast/clblast.cpp ${RNLLAMA_LIB_DIR}/CLBlast/clblast_c.cpp ${RNLLAMA_LIB_DIR}/CLBlast/tuning/tuning_api.cpp)
set(OPENCL_HEADERS ${OPENCL_HEADERS} ${RNLLAMA_LIB_DIR}/CLBlast/clblast.h ${RNLLAMA_LIB_DIR}/CLBlast/clblast_c.h ${RNLLAMA_LIB_DIR}/CLBlast/clpp11.hpp)

foreach(ROUTINE ${LEVEL1_ROUTINES})
  set(OPENCL_SOURCES ${OPENCL_SOURCES} ${RNLLAMA_LIB_DIR}/CLBlast/routines/level1/${ROUTINE}.cpp)
  set(OPENCL_HEADERS ${OPENCL_HEADERS} ${RNLLAMA_LIB_DIR}/CLBlast/routines/level1/${ROUTINE}.hpp)
endforeach()
foreach(ROUTINE ${LEVEL2_ROUTINES})
  set(OPENCL_SOURCES ${OPENCL_SOURCES} ${RNLLAMA_LIB_DIR}/CLBlast/routines/level2/${ROUTINE}.cpp)
  set(OPENCL_HEADERS ${OPENCL_HEADERS} ${RNLLAMA_LIB_DIR}/CLBlast/routines/level2/${ROUTINE}.hpp)
endforeach()
foreach(ROUTINE ${LEVEL3_ROUTINES})
  set(OPENCL_SOURCES ${OPENCL_SOURCES} ${RNLLAMA_LIB_DIR}/CLBlast/routines/level3/${ROUTINE}.cpp)
  set(OPENCL_HEADERS ${OPENCL_HEADERS} ${RNLLAMA_LIB_DIR}/CLBlast/routines/level3/${ROUTINE}.hpp)
endforeach()
foreach(ROUTINE ${LEVELX_ROUTINES})
  set(OPENCL_SOURCES ${OPENCL_SOURCES} ${RNLLAMA_LIB_DIR}/CLBlast/routines/levelx/${ROUTINE}.cpp)
  set(OPENCL_HEADERS ${OPENCL_HEADERS} ${RNLLAMA_LIB_DIR}/CLBlast/routines/levelx/${ROUTINE}.hpp)
endforeach()
foreach(DATABASE ${DATABASES})
  set(OPENCL_SOURCES ${OPENCL_SOURCES} ${RNLLAMA_LIB_DIR}/CLBlast/database/kernels/${DATABASE}/${DATABASE}.cpp)
  set(OPENCL_HEADERS ${OPENCL_HEADERS} ${RNLLAMA_LIB_DIR}/CLBlast/database/kernels/${DATABASE}/${DATABASE}.hpp)
  set(OPENCL_HEADERS ${OPENCL_HEADERS} ${RNLLAMA_LIB_DIR}/CLBlast/database/kernels/${DATABASE}/${DATABASE}_16.hpp)
  set(OPENCL_HEADERS ${OPENCL_HEADERS} ${RNLLAMA_LIB_DIR}/CLBlast/database/kernels/${DATABASE}/${DATABASE}_32.hpp)
  set(OPENCL_HEADERS ${OPENCL_HEADERS} ${RNLLAMA_LIB_DIR}/CLBlast/database/kernels/${DATABASE}/${DATABASE}_64.hpp)
  set(OPENCL_HEADERS ${OPENCL_HEADERS} ${RNLLAMA_LIB_DIR}/CLBlast/database/kernels/${DATABASE}/${DATABASE}_3232.hpp)
  set(OPENCL_HEADERS ${OPENCL_HEADERS} ${RNLLAMA_LIB_DIR}/CLBlast/database/kernels/${DATABASE}/${DATABASE}_6464.hpp)
endforeach()
foreach(KERNEL ${KERNELS})
  set(OPENCL_HEADERS ${OPENCL_HEADERS} ${RNLLAMA_LIB_DIR}/CLBlast/tuning/kernels/${KERNEL}.hpp)
endforeach()

set(RNLLAMA_LIBRARY_NAME rnllama)

add_library(
    ${RNLLAMA_LIBRARY_NAME}
    SHARED
    ${SOURCE_FILES}
    ${OPENCL_SOURCES}
    ${OPENCL_HEADERS}
)


    target_link_libraries(${target_name} ${LOG_LIB} android)


# Build stub
add_subdirectory(libopencl-stub)

# set(libs "${CMAKE_SOURCE_DIR}/jniLibs")

# add_library(libOpenCL SHARED IMPORTED)
# set_target_properties(libOpenCL PROPERTIES
#     IMPORTED_LOCATION "${libs}/${ANDROID_ABI}/libOpenCL.so")

target_link_libraries(${RNLLAMA_LIBRARY_NAME} ${LOG_LIB} android OpenCL)
include_directories(${RNLLAMA_LIB_DIR})
# include_directories(${RNLLAMA_LIB_DIR}/CL)
include_directories(${RNLLAMA_LIB_DIR}/CLBlast)

target_include_directories(${RNLLAMA_LIBRARY_NAME} PRIVATE
  libopencl-stub/include/
)

target_compile_options(${RNLLAMA_LIBRARY_NAME} PRIVATE -DLM_GGML_USE_K_QUANTS -DLM_GGML_USE_CLBLAST -DUSE_STUB_OPENCL -DOPENCL_API)


    target_compile_options(${target_name} PRIVATE -O3 -DNDEBUG)
    target_compile_options(${target_name} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
    target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)


target_compile_options(${RNLLAMA_LIBRARY_NAME} PRIVATE -O3 -DNDEBUG)
target_compile_options(${RNLLAMA_LIBRARY_NAME} PRIVATE -fvisibility=hidden -fvisibility-inlines-hidden)
target_compile_options(${RNLLAMA_LIBRARY_NAME} PRIVATE -ffunction-sections -fdata-sections)


    # endif ()
endfunction()

build_library("rnllama") # Default target

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    build_library("rnllama_v8fp16_va")
endif ()
