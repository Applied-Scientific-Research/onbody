#
# onbody
#
# (c)2017-24 Applied Scientific Research, Inc.
#
CMAKE_MINIMUM_REQUIRED( VERSION 3.11 )
PROJECT( onbody CXX )

INCLUDE( FetchContent )

SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
IF (NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE "Release")
ENDIF ()
SET (CMAKE_BUILD_TYPE "Release" CACHE STRING "Select which configuration to build")
SET (ONBODY_USE_OMP TRUE CACHE BOOL "Use OpenMP multithreading")
SET (ONBODY_USE_VC TRUE CACHE BOOL "Use Vc for vector arithmetic")
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
  # which also requires this
  SET (EXTRA_LINK "-static-libstdc++")

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

# 3rd party libraries
SET (ONBODY_LIBS "")

# required libraries
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
SET( ONBODY_LIBS ${ONBODY_LIBS} Eigen3::Eigen )

# optional libraries
FetchContent_Declare(
  Vc
  URL https://github.com/VcDevel/Vc/archive/refs/tags/1.4.5.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
IF( ONBODY_USE_VC )
  FetchContent_MakeAvailable(Vc)
ENDIF ()

# OpenMP for multithreading
IF( ONBODY_USE_OMP )
  FIND_PACKAGE (OpenMP REQUIRED)
  IF (OPENMP_FOUND)
    SET( ONBODY_LIBS ${ONBODY_LIBS} OpenMP::OpenMP_CXX )
  ENDIF ()
ENDIF()

# Split on platform

IF ( WIN32 )

  SET ( PLATFORM_LIBS ${FRAMEWORK_LIBS} )

  SET( ONBODY_USE_VC FALSE )

ELSEIF ( APPLE )

  INCLUDE_DIRECTORIES ( /System/Library/Frameworks )
  FIND_LIBRARY (COCOA_LIBRARY Cocoa)
  FIND_LIBRARY (IOKIT_LIBRARY IOKit)
  MARK_AS_ADVANCED (COCOA_LIBRARY IOKIT_LIBRARY)

  SET (PLATFORM_LIBS ${FRAMEWORK_LIBS} ${COCOA_LIBRARY} ${IOKIT_LIBRARY})

  SET( ONBODY_USE_VC FALSE )

ELSEIF ( UNIX )

  # OpenMP for multithreading
  IF( ONBODY_USE_OMP )
  ELSE()
    # suppress the warning
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    # need to find pthreads (for std::async?)
    FIND_PACKAGE( Threads REQUIRED )
  ENDIF()

  # Vc for explicit vectorization
  IF( ONBODY_USE_VC )
    SET (CPREPROCDEFS ${CPREPROCDEFS} -DUSE_VC)
    SET( ONBODY_LIBS ${ONBODY_LIBS} Vc )
  ENDIF()

ENDIF ()

INCLUDE_DIRECTORIES ( "src" )
ADD_DEFINITIONS (${CPREPROCDEFS})

# main applications
ADD_EXECUTABLE ( "onvort2d" "src/onvort2d.cpp" )
TARGET_LINK_LIBRARIES( "onvort2d" ${ONBODY_LIBS} )
SET_TARGET_PROPERTIES( "onvort2d" PROPERTIES LINK_FLAGS ${EXTRA_LINK})

ADD_EXECUTABLE ( "onvort3d" "src/onvort3d.cpp" )
TARGET_LINK_LIBRARIES( "onvort3d" ${ONBODY_LIBS} )
SET_TARGET_PROPERTIES( "onvort3d" PROPERTIES LINK_FLAGS ${EXTRA_LINK})

ADD_EXECUTABLE ( "ongrav3d" "src/ongrav3d.cpp" )
TARGET_LINK_LIBRARIES( "ongrav3d" ${ONBODY_LIBS} )
SET_TARGET_PROPERTIES( "ongrav3d" PROPERTIES LINK_FLAGS ${EXTRA_LINK})

ADD_EXECUTABLE ( "onvortgrad3d" "src/onvortgrad3d.cpp" )
TARGET_LINK_LIBRARIES( "onvortgrad3d" ${ONBODY_LIBS} )
SET_TARGET_PROPERTIES( "onvortgrad3d" PROPERTIES LINK_FLAGS ${EXTRA_LINK})

# linkable libraries and test programs
ADD_LIBRARY ( "bh2dvort" STATIC "src/interface2dvort.cpp" "src/interface2dvorttr.cpp" )
TARGET_INCLUDE_DIRECTORIES( "bh2dvort" PUBLIC "src" )
TARGET_LINK_LIBRARIES( "bh2dvort" ${ONBODY_LIBS} )

ADD_EXECUTABLE ( "run2dvort" "src/main2dvort.cpp" )
TARGET_LINK_LIBRARIES ( "run2dvort" "bh2dvort" ${ONBODY_LIBS} )
SET_TARGET_PROPERTIES( "run2dvort" PROPERTIES LINK_FLAGS ${EXTRA_LINK})

ADD_LIBRARY ( "bh3dvortgrads" STATIC "src/interface3dvortgrads.cpp" )
TARGET_INCLUDE_DIRECTORIES( "bh3dvortgrads" PUBLIC "src" )
TARGET_LINK_LIBRARIES( "bh3dvortgrads" ${ONBODY_LIBS} )

ADD_EXECUTABLE ( "run3dvortgrads" "src/main3dvortgrads.cpp" )
TARGET_LINK_LIBRARIES ( "run3dvortgrads" "bh3dvortgrads" ${ONBODY_LIBS} )
SET_TARGET_PROPERTIES( "run3dvortgrads" PROPERTIES LINK_FLAGS ${EXTRA_LINK})

