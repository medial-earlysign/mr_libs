cmake_minimum_required(VERSION 3.10)
project(medpython)
cmake_policy(SET CMP0074 NEW)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.27")
    cmake_policy(SET CMP0144 NEW)
endif()
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.13")
    cmake_policy(SET CMP0077 NEW)
endif()

if(UNIX AND NOT APPLE)
    include(CheckIncludeFileCXX)
    
    # Check if execinfo.h exists (Standard Linux/glibc has it, Alpine/Musl does not)
    check_include_file_cxx("execinfo.h" HAVE_EXECINFO)

    if(NOT HAVE_EXECINFO)
        message(STATUS "Musl libc (Alpine) detected. Applying compatibility flags.")
        
        # Add the flags specifically for this environment
        add_compile_definitions(
            DMLC_LOG_STACK_TRACE=0
            DMLC_CMAKE_LITTLE_ENDIAN=1
            mmap64=mmap
            SO_COMPILATION
        )
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Boost
find_package(OpenMP REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/Linux/${CMAKE_BUILD_TYPE}")
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR})

if(DEFINED ENV{BOOST_ROOT})
    message(STATUS "BOOST_ROOT is set to: $ENV{BOOST_ROOT}")
    set(BOOST_ROOT "$ENV{BOOST_ROOT}")
    set(Boost_INCLUDE_DIRS "${BOOST_ROOT}/include")
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
else()
    find_package(Boost QUIET)
    if (NOT Boost_FOUND)
        message(STATUS "BOOST_ROOT environment variable is not set. Please set it to the Boost installation path, if you want to use a specific boost.")
        message(STATUS "Boost not found. Downloading headers...")
        include(FetchContent)
        FetchContent_Declare(
            boost_headers
                URL      https://archives.boost.io/release/1.89.0/source/boost_1_89_0.tar.gz
                SOURCE_SUBDIR pathThatDoesNotExist
                DOWNLOAD_EXTRACT_TIMESTAMP ON
            )
            
            FetchContent_GetProperties(boost_headers)
            if(NOT boost_headers_POPULATED)
                FetchContent_MakeAvailable(boost_headers)
            endif()

            include_directories(${boost_headers_SOURCE_DIR})
    endif()
endif()

set (MEDIAL_INTERNAL_LIBS InfraMed Logger MedAlgo MedEmbed MedIO MedMat MedPlotly MedProcessTools MedSparseMat MedSplit MedStat MedTime MedUtils QRF SerializableObject TQRF micNet CommonLib
)

set(BUILD_STATIC_LIB ON CACHE BOOL "Build XGBoost as a static library" FORCE)
if (NOT WIN32)
    # 1. Base flags common to all Unix/Linux/Mac systems
    set(COMMON_FLAGS "-fPIC -Wno-write-strings -Wuninitialized -Wno-narrowing -DMES_LIBRARY=1 -DGIT_HEAD_VERSION='\"'$(GIT_HEAD_VERSION)'\"'")

    if(APPLE)
        set(COMPILER_FLAGS "${COMMON_FLAGS} -Xpreprocessor -fopenmp -D_LIBCPP_DISABLE_AVAILABILITY")
    else()
        # Linux specific base flags
        set(COMPILER_FLAGS "${COMMON_FLAGS} -fopenmp --param inline-unit-growth=1000000")

        # 2. Check Architecture
        # Matches aarch64 (Linux) or arm64 (Apple Silicon)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
            message(STATUS "Target architecture: x86_64")
            set(COMPILER_FLAGS "${COMPILER_FLAGS} -msse2 -msse3 -msse4 -march=x86-64")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i686)$")
            message(STATUS "Detected x86 (32-bit) architecture - i686")
            set(COMPILER_FLAGS "${COMPILER_FLAGS} -msse2 -march=i686")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i386|x86)$")
            message(STATUS "Detected x86 (32-bit) architecture")
            set(COMPILER_FLAGS "${COMPILER_FLAGS} -march=i386")
        else()
            message(STATUS "Using Generic Flags To Target architecture: ${CMAKE_SYSTEM_PROCESSOR}")
        endif()
    endif()
else()
    # Windows
    set(COMPILER_FLAGS "/openmp /EHsc /MD /permissive- /Zc:strictStrings- /Zc:__cplusplus /DMES_LIBRARY=1")
    add_compile_definitions(GIT_HEAD_VERSION="$ENV{GIT_HEAD_VERSION}")
    add_compile_definitions(BOOST_ALL_NO_LIB)
endif()
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${COMPILER_FLAGS} ")
# Filter out definition of NDEBUG from the default configuration flags for Release
foreach (language CXX C)
	set(VAR_TO_MODIFY "CMAKE_${language}_FLAGS_RELEASE")
	string(REGEX REPLACE "(^| )[/-]D *NDEBUG($| )"
						 " "
						 replacement
						 "${${VAR_TO_MODIFY}}"
		  )
	message("Original (${VAR_TO_MODIFY}) is ${${VAR_TO_MODIFY}} replacement is ${replacement}")
	set(${VAR_TO_MODIFY} "${replacement}" CACHE STRING "Default flags for Release configuration" FORCE)
endforeach()


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -march=native -Og")

set(CMAKE_BUILD_TYPE "Release")


set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../External)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../Internal)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../External/xgboost/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../External/xgboost/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../External/xgboost/rabit/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../External/xgboost/dmlc-core/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../External/LightGBM_2.2.3/LightGBM-2.2.3/include)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libs" FORCE)
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL" CACHE STRING "Runtime library" FORCE)

# Add Internal libraries
foreach(mes_lib IN LISTS MEDIAL_INTERNAL_LIBS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../${mes_lib}/${mes_lib} ${mes_lib})
endforeach()

include_directories(../MedPyExport)

# External Libraries
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../External/xgboost xgboost)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../External/LightGBM_2.2.3/LightGBM-2.2.3 _lightgbm)

add_subdirectory(MedPython)