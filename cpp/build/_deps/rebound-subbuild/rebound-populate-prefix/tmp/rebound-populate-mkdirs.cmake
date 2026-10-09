# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "D:/iris_calibration/cpp/build/_deps/rebound-src")
  file(MAKE_DIRECTORY "D:/iris_calibration/cpp/build/_deps/rebound-src")
endif()
file(MAKE_DIRECTORY
  "D:/iris_calibration/cpp/build/_deps/rebound-build"
  "D:/iris_calibration/cpp/build/_deps/rebound-subbuild/rebound-populate-prefix"
  "D:/iris_calibration/cpp/build/_deps/rebound-subbuild/rebound-populate-prefix/tmp"
  "D:/iris_calibration/cpp/build/_deps/rebound-subbuild/rebound-populate-prefix/src/rebound-populate-stamp"
  "D:/iris_calibration/cpp/build/_deps/rebound-subbuild/rebound-populate-prefix/src"
  "D:/iris_calibration/cpp/build/_deps/rebound-subbuild/rebound-populate-prefix/src/rebound-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/iris_calibration/cpp/build/_deps/rebound-subbuild/rebound-populate-prefix/src/rebound-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/iris_calibration/cpp/build/_deps/rebound-subbuild/rebound-populate-prefix/src/rebound-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
