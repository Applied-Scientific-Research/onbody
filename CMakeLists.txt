#
# onbody
#
# (c)2017-2025 Applied Scientific Research, Inc.
#
CMAKE_MINIMUM_REQUIRED( VERSION 3.5 )
PROJECT( onbody LANGUAGES CXX )

SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
IF (NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE "Release")
ENDIF ()
SET (CMAKE_BUILD_TYPE "Release" CACHE STRING "Select which configuration to build")
SET (ONBODY_USE_OMP FALSE CACHE BOOL "Use OpenMP multithreading")
SET (ONBODY_USE_VC FALSE CACHE BOOL "Use Vc for vector arithmetic")
SET (ONBODY_PROFILE FALSE CACHE BOOL "Enable instrumentation of user functions")
SET (CMAKE_VERBOSE_MAKEFILE on)
SET (CMAKE_EXPORT_COMPILE_COMMANDS on)

IF (NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE "Release")
ENDIF ()
SET (EXTRA_LINK "")

IF (CMAKE_COMPILER_IS_GNUCXX)
  SET (CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wformat -std=c++14")
  SET (CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb")
  SET (CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -flto")
  SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=native -ffast-math -flto -g -ggdb3")
  # need the following for -flto to work
  # see https://stackoverflow.com/questions/39236917/using-gccs-link-time-optimization-with-static-linked-libraries
  SET (CMAKE_AR "gcc-ar")
  SET (CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qcs <TARGET> <LINK_FLAGS> <OBJECTS>")
  SET (CMAKE_CXX_ARCHIVE_FINISH true)
  # which also requires this, but not gcc 14.2.1
  #SET (EXTRA_LINK "-static-libstdc++")

ELSEIF (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # The Cray compiler reports as Clang to most versions of CMake
  EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                  COMMAND grep -c Cray
                  OUTPUT_VARIABLE INTERNAL_HAVE_CRAY_COMPILER
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  IF (INTERNAL_HAVE_CRAY_COMPILER) #not actually Clang
    # cray compiler misidentifies as Clang and needs this instead
    SET (CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wformat -std=c++14")
  ELSE()
    # real clang needs this
    SET (CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wformat -std=c++14 -stdlib=libc++")
  ENDIF()
  SET (CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb")
  SET (CMAKE_CXX_FLAGS_RELEASE "-Ofast -march=native -ffast-math -flto")
  SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -ggdb")

ELSEIF (MSVC)
  SET (CMAKE_CXX_FLAGS "/Ox")
ENDIF ()

IF ( WIN32 )

  SET ( PLATFORM_LIBS ${FRAMEWORK_LIBS} )

ELSEIF ( APPLE )

  INCLUDE_DIRECTORIES ( /System/Library/Frameworks )
  FIND_LIBRARY (COCOA_LIBRARY Cocoa)
  FIND_LIBRARY (IOKIT_LIBRARY IOKit)
  MARK_AS_ADVANCED (COCOA_LIBRARY IOKIT_LIBRARY)

  SET (PLATFORM_LIBS ${FRAMEWORK_LIBS} ${COCOA_LIBRARY} ${IOKIT_LIBRARY})

ELSEIF ( UNIX )

  SET (EIGEN_ROOT "/usr/include/eigen3" CACHE STRING "Eigen root" )
  INCLUDE_DIRECTORIES ( "${EIGEN_ROOT}" )

ENDIF ()

IF( ONBODY_PROFILE )
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -finstrument-functions -fprofile-exclude-files=/usr/.* -finstrument-functions-exclude-file-list=include")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -finstrument-functions -fprofile-exclude-files=/usr/.* -finstrument-functions-exclude-file-list=include")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic -finstrument-functions -fprofile-exclude-files=/usr/.* -finstrument-functions-exclude-file-list=include")
ENDIF ()

INCLUDE_DIRECTORIES ( "src" )
ADD_DEFINITIONS (${CPREPROCDEFS})

IF( NOT ONBODY_USE_OMP )
  # silence the warnings
  # this line must appear BEFORE defining the executables
  ADD_COMPILE_OPTIONS( -Wno-unknown-pragmas )
  # need to find pthreads (for std::async?)
  FIND_PACKAGE( Threads REQUIRED )
ENDIF()

# original targets
ADD_EXECUTABLE ( "onvort2d" "src/onvort2d.cpp" )
ADD_EXECUTABLE ( "onvort3d" "src/onvort3d.cpp" )
ADD_EXECUTABLE ( "ongrav3d" "src/ongrav3d.cpp" )
ADD_EXECUTABLE ( "onvortgrad3d" "src/onvortgrad3d.cpp" )

ADD_LIBRARY ( "bh2dvort" STATIC "src/interface2dvort.cpp" "src/interface2dvorttr.cpp" )
ADD_EXECUTABLE ( "run2dvort" "src/main2dvort.cpp" )
TARGET_LINK_LIBRARIES( "run2dvort" PUBLIC "bh2dvort")

ADD_LIBRARY ( "bh3dvortgrads" STATIC "src/interface3dvortgrads.cpp" )
ADD_EXECUTABLE ( "run3dvortgrads" "src/main3dvortgrads.cpp" )
TARGET_LINK_LIBRARIES( "run3dvortgrads" PUBLIC "bh3dvortgrads")

# OpenMP for multithreading
IF( ONBODY_USE_OMP )
  FIND_PACKAGE( OpenMP REQUIRED )
  TARGET_LINK_LIBRARIES( "onvort2d" PUBLIC OpenMP::OpenMP_CXX)
  TARGET_LINK_LIBRARIES( "onvort3d" PUBLIC OpenMP::OpenMP_CXX)
  TARGET_LINK_LIBRARIES( "onvortgrad3d" PUBLIC OpenMP::OpenMP_CXX)
  TARGET_LINK_LIBRARIES( "ongrav3d" PUBLIC OpenMP::OpenMP_CXX)
  TARGET_LINK_LIBRARIES( "bh2dvort" PUBLIC OpenMP::OpenMP_CXX)
  TARGET_LINK_LIBRARIES( "run2dvort" PUBLIC OpenMP::OpenMP_CXX)
  TARGET_LINK_LIBRARIES( "bh3dvortgrads" PUBLIC OpenMP::OpenMP_CXX)
  TARGET_LINK_LIBRARIES( "run3dvortgrads" PUBLIC OpenMP::OpenMP_CXX)
  ADD_COMPILE_OPTIONS( ${OpenMP_CXX_FLAGS} )
ENDIF()  # OpenMP for multithreading

# Vc for explicit vectorization
IF( ONBODY_USE_VC )
  FIND_PACKAGE( Vc REQUIRED )
  ADD_DEFINITIONS(-DUSE_VC)
  INCLUDE_DIRECTORIES( ${Vc_INCLUDE_DIR} )
  # these lines must appear AFTER defining the executables
  TARGET_LINK_LIBRARIES( "onvort2d" PUBLIC ${Vc_LIBRARIES} )
  TARGET_LINK_LIBRARIES( "onvort3d" PUBLIC ${Vc_LIBRARIES} )
  TARGET_LINK_LIBRARIES( "onvortgrad3d" PUBLIC ${Vc_LIBRARIES} )
  TARGET_LINK_LIBRARIES( "ongrav3d" PUBLIC ${Vc_LIBRARIES} )
  TARGET_LINK_LIBRARIES( "bh2dvort" PUBLIC ${Vc_LIBRARIES} )
  TARGET_LINK_LIBRARIES( "run2dvort" PUBLIC ${Vc_LIBRARIES} )
  TARGET_LINK_LIBRARIES( "bh3dvortgrads" PUBLIC ${Vc_LIBRARIES} )
  TARGET_LINK_LIBRARIES( "run3dvortgrads" PUBLIC ${Vc_LIBRARIES} )
ENDIF()  # Vc for explicit vectorization

