@echo off
REM Build and run the direct C++ kernel test.
REM Requires the TensorRT runtime DLLs on PATH at run time.

setlocal
set "ROOT=%~dp0.."
set "VS=C:\Program Files\Microsoft Visual Studio\2022\Professional"

call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo [ERROR] vcvars64 failed & exit /b 1 )

set "TRT=C:\TensorRT-10.16.0.72"
set "CUDA=C:\CUDA"

cl.exe /nologo /EHsc /std:c++17 /O2 /utf-8 /wd4819 ^
  /I"%ROOT%\include" /I"%TRT%\include" /I"%CUDA%\include" ^
  "%ROOT%\tests\test_kernel.cpp" ^
  /link /LIBPATH:"%ROOT%\build\Release" /LIBPATH:"%TRT%\lib" /LIBPATH:"%CUDA%\lib\x64" ^
  qualityscaler_tensorrt.lib nvinfer_10.lib cudart.lib ^
  /OUT:"%ROOT%\tests\test_kernel.exe"
if errorlevel 1 ( echo [ERROR] compile failed & exit /b 1 )

echo [OK] built test_kernel.exe
REM The kernel DLL and the TensorRT/CUDA runtime DLLs must all be findable.
set "PATH=%ROOT%\build\bin\Release;%ROOT%\build\Release;%TRT%\bin;%CUDA%\bin;%PATH%"
"%ROOT%\tests\test_kernel.exe"
exit /b %ERRORLEVEL%
