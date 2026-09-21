@echo off
setlocal enabledelayedexpansion

REM ==================================================================
REM  build_package.bat
REM
REM  1) Builds the C# GpuAccess library from package_factory\
REM  2) Copies the compiled DLLs into the Python package
REM  3) Builds an installable Python wheel (cugpu_pkg\dist\*.whl)
REM
REM  Requirements on this machine:
REM    - .NET SDK 8.0+   (https://dotnet.microsoft.com/download)
REM    - Python 3.8+ on PATH
REM ==================================================================

set "ROOT=%~dp0"
set "CSHARP_DIR=%ROOT%package_factory"
set "PY_PKG_DIR=%ROOT%cugpu_pkg"
set "NATIVE_DIR=%PY_PKG_DIR%\cugpu\_native"
set "CONFIG=Release"
set "TFM=net8.0"

echo.
echo === [1/6] Checking prerequisites ===
where dotnet >nul 2>nul
if errorlevel 1 (
    echo ERROR: .NET SDK not found on PATH. Install it from https://dotnet.microsoft.com/download
    exit /b 1
)
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python not found on PATH.
    exit /b 1
)

echo.
echo === [2/6] Ensuring Python package structure and source files exist ===
python "%ROOT%generate_pkg.py"
if errorlevel 1 (
    echo ERROR: Failed to generate/verify Python package structure.
    exit /b 1
)

echo.
echo === [3/6] Building C# GpuAccess library (package_factory) ===
dotnet build "%CSHARP_DIR%\GpuAccess.csproj" -c %CONFIG%
if errorlevel 1 (
    echo ERROR: dotnet build failed.
    exit /b 1
)

echo.
echo === [4/6] Copying compiled assemblies into the Python package ===
if not exist "%NATIVE_DIR%" mkdir "%NATIVE_DIR%"

set "BIN_DIR=%CSHARP_DIR%\bin\%CONFIG%\%TFM%"
if not exist "%BIN_DIR%" (
    echo ERROR: Expected build output folder not found: %BIN_DIR%
    echo        Check that TFM/CONFIG at the top of this script match your .csproj.
    exit /b 1
)

copy /Y "%BIN_DIR%\*.dll" "%NATIVE_DIR%\" >nul
echo Copied DLLs from:
echo   %BIN_DIR%
echo to:
echo   %NATIVE_DIR%

echo.
echo === [5/6] Installing Python build tooling ===
python -m pip install --upgrade pip build >nul
if errorlevel 1 (
    echo ERROR: Failed to install/upgrade the 'build' package.
    exit /b 1
)

echo.
echo === [6/6] Building the Python wheel ===
pushd "%PY_PKG_DIR%"
python -m build
if errorlevel 1 (
    popd
    echo ERROR: python -m build failed.
    exit /b 1
)
popd

echo.
echo === DONE ===
echo Wheel and sdist were written to: %PY_PKG_DIR%\dist
echo.
echo Install the package with, e.g.:
echo   pip install "%PY_PKG_DIR%\dist\cugpu-0.1.0-py3-none-any.whl"
echo.
echo Then in Python:
echo   import cugpu
echo   cugpu.print_gpus()
echo.

endlocal
