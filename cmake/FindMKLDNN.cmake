#
# Find mkldnn
#
#  MKLDNN_INCLUDE_DIR - where to find mkldnn.hpp, mkldnn.h, mkldnn_type.h
#  MKLDNN_LIBRARY     - List of libraries when using libmkldnn.so
#  MKLDNN_FOUND       - True if mkldnn found.
#
# User should set one of MKLDNN_ROOT during cmake. 
# If none of them set, it will try to find mkldnn implementation in system paths.
# 
#

if (MKLDNN_LIBRARY)
  # Already in cache, be silent
  set(MKLDNN_FIND_QUIETLY TRUE)
endif ()

## Find MKLDNN
set(MKLDNN_ROOT $ENV{MKLDNN_ROOT} CACHE PATH "Folder contains DNN.....................")

find_path(MKLDNN_INCLUDE_DIR mkldnn.hpp PATHS ${MKLDNN_ROOT}/include 
    /usr/local/include)
find_library(MKLDNN_LIBRARY NAMES mkldnn PATHS ${MKLDNN_ROOT}/lib
    /usr/local/lib)

# handle the QUIETLY and REQUIRED arguments and set MKLDNN_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MKLDNN DEFAULT_MSG MKLDNN_LIBRARY MKLDNN_INCLUDE_DIR)

MARK_AS_ADVANCED(MKLDNN_LIBRARY MKLDNN_INCLUDE_DIR)

