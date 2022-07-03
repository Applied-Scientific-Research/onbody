#
# onbody
# (c)2017-2022 Applied Scientific Research, Inc.
#
CMAKE_MINIMUM_REQUIRED ( VERSION 2.6 )
PROJECT ( onbody )

# User-serviceable variables here
SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
IF (NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE "Release")
ENDIF ()
SET (CMAKE_BUILD_TYPE "Release" CACHE STRING "Select which configuration to build")
SET (USE_OMP FALSE CACHE BOOL "Use OpenMP multithreading")
SET (USE_VC FALSE CACHE BOOL "Use Vc for vector arithmetic")
SET (CMAKE_VERBOSE_MAKEFILE on)
SET (CMAKE_EXPORT_COMPILE_COMMANDS on)

IF (NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE "Release")
ENDIF ()

IF (CMAKE_COMPILER_IS_GNUCXX)
  SET (CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wformat -std=c++14")
  SET (CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb")
  SET (CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")
  SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=native -g -ggdb3")
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
  SET (CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")
  SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -ggdb")
ELSEIF (MSVC)
  SET (CMAKE_CXX_FLAGS "/Ox")
ENDIF ()

IF ( WIN32 )

  SET ( PLATFORM_LIBS ${FRAMEWORK_LIBS} )

  # OpenMP for multithreading
  IF( USE_OMP )
    FIND_PACKAGE (OpenMP REQUIRED)
    IF (OPENMP_FOUND)
      #MESSAGE ("OPENMP FOUND")
      SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    ENDIF ()
  ENDIF()

ELSEIF ( APPLE )

  INCLUDE_DIRECTORIES ( /System/Library/Frameworks )
  FIND_LIBRARY (COCOA_LIBRARY Cocoa)
  FIND_LIBRARY (IOKIT_LIBRARY IOKit)
  MARK_AS_ADVANCED (COCOA_LIBRARY IOKIT_LIBRARY)

  SET (PLATFORM_LIBS ${FRAMEWORK_LIBS} ${COCOA_LIBRARY} ${IOKIT_LIBRARY})

  # OpenMP for multithreading
  IF( USE_OMP )
    FIND_PACKAGE (OpenMP REQUIRED)
    IF (OPENMP_FOUND)
      #MESSAGE ("OPENMP FOUND")
      SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    ENDIF ()
  ENDIF()

ELSEIF ( UNIX )

  # OpenMP for multithreading
  IF( USE_OMP )
    FIND_PACKAGE( OpenMP REQUIRED )
    IF( OPENMP_FOUND )
      #message("OPENMP FOUND")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    ENDIF()
  ELSE()
    # suppress the warning
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    # need to find pthreads (for std::async?)
    FIND_PACKAGE( Threads REQUIRED )
  ENDIF()

  # Vc for explicit vectorization
  IF( USE_VC )
    SET (CPREPROCDEFS ${CPREPROCDEFS} -DUSE_VC)
    SET (VC_ROOT "/opt/Vc" CACHE STRING "Vc root" )
    INCLUDE_DIRECTORIES( "${VC_ROOT}/include" )
    FIND_LIBRARY( VC_LIBS NAMES Vc PATHS "${VC_ROOT}/lib" )
  ELSE()
    SET (VC_LIBS "")
  ENDIF()

  SET (EIGEN_ROOT "/usr/include/eigen3" CACHE STRING "Eigen root" )
  INCLUDE_DIRECTORIES ( "${EIGEN_ROOT}" )

ENDIF ()

INCLUDE_DIRECTORIES ( "src" )
ADD_DEFINITIONS (${CPREPROCDEFS})

# original targets
ADD_EXECUTABLE ( "onvort2d" "src/onvort2d.cpp" )
TARGET_LINK_LIBRARIES( "onvort2d" ${VC_LIBS} )

ADD_EXECUTABLE ( "onvort3d" "src/onvort3d.cpp" )
TARGET_LINK_LIBRARIES( "onvort3d" ${VC_LIBS} )

ADD_EXECUTABLE ( "ongrav3d" "src/ongrav3d.cpp" )
TARGET_LINK_LIBRARIES( "ongrav3d" ${VC_LIBS} )

ADD_EXECUTABLE ( "onvortgrad3d" "src/onvortgrad3d.cpp" )
TARGET_LINK_LIBRARIES( "onvortgrad3d" ${VC_LIBS} )

# new targets
ADD_LIBRARY ( "bh2dvort" STATIC "src/interface2dvort.cpp" "src/interface2dvorttr.cpp" )
ADD_EXECUTABLE ( "run2dvort" "src/main2dvort.cpp" )
TARGET_LINK_LIBRARIES ( "run2dvort" "bh2dvort" ${VC_LIBS} )

ADD_LIBRARY ( "bh3dvortgrads" STATIC "src/interface3dvortgrads.cpp" )
ADD_EXECUTABLE ( "run3dvortgrads" "src/main3dvortgrads.cpp" )
TARGET_LINK_LIBRARIES ( "run3dvortgrads" "bh3dvortgrads" ${VC_LIBS} )

#TARGET_LINK_LIBRARIES ( ${PROJECT_NAME} ${PLATFORM_LIBS} )

