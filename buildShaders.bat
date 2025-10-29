REM buidlShaders.bat to allow premake to use a buildcommand
REM to complile all shaders everytime the project is built,
REM instead of only recompiling changed shaders that Visual
REM Studio detects, this makes it easier to have working shaders
REM when using #includes in shader files which Visual Studio
REM does not track the dependency of.
@echo off
setlocal enabledelayedexpansion

REM Usage: build_shaders.bat [Debug] or [Release]
set BUILD_CONFIG=%1
if "%BUILD_CONFIG%"=="" set BUILD_CONFIG=Debug

REM Paths
set SHADERC_DIR=%~dp0third_party\shaderc
set OUTPUT_DIR=%~dp0assets\main\shaders
set GLSLC_BIN=%SHADERC_DIR%\win-x86_64\glslc.exe

if not exist "%GLSLC_BIN%" (
    echo glslc not found at "%GLSLC_BIN%"
    exit /b 1
)

REM GLSLC options based on passed argument configuration
set GLSLC_OPTS=-O --target-env=vulkan1.2 -w
if /I "%BUILD_CONFIG%"=="Debug" set GLSLC_OPTS=%GLSLC_OPTS% -g

REM Shader types to compile
set SHADER_EXTS=vert frag comp geom tesc tese

REM Compile shaders
echo [GLSLC] Compiling shaders
for %%E in (%SHADER_EXTS%) do (
    for %%F in (shaders\*.%%E) do (
        if exist "%%F" (
            set "OUT_FILE=%OUTPUT_DIR%\%%~nxF.spv"
            if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
            echo [GLSLC] Compiling %%F -> !OUT_FILE!
            "%GLSLC_BIN%" %GLSLC_OPTS% -o "!OUT_FILE!" "%%F"
        )
    )
)

endlocal
