workspace "VulkanRenderer"
	language "C++"
	cppdialect "C++20"

	platforms { "x64" }
	configurations { "debug", "release" }

	flags "NoPCH"
	flags "MultiProcessorCompile"

	startproject "main"

	debugdir "%{wks.location}"
	objdir "_build_/%{cfg.buildcfg}-%{cfg.platform}-%{cfg.toolset}"
	targetsuffix "-%{cfg.buildcfg}-%{cfg.platform}-%{cfg.toolset}"
	
	-- Default toolset options
	filter "toolset:gcc or toolset:clang"
		linkoptions { "-pthread" }
		buildoptions { "-march=native", "-Wall", "-pthread" }

	filter "toolset:msc-*"
		defines { "_CRT_SECURE_NO_WARNINGS=1" }
		defines { "_SCL_SECURE_NO_WARNINGS=1" }
		buildoptions { "/utf-8" }
	
	filter "*"

	-- default options for GLSLC
	glslcOptions = "-O --target-env=vulkan1.2"

	-- default libraries
	filter "system:linux"
		links "dl"
	
	filter "system:windows"

	filter "*"

	-- default outputs
	filter "kind:StaticLib"
		targetdir "lib/"

	filter "kind:ConsoleApp"
		targetdir "bin/"
		targetextension ".exe"
	
	filter "*"

	--configurations
	filter "debug"
		symbols "On"
		defines { "_DEBUG=1" }

	filter "release"
		optimize "On"
		defines { "NDEBUG=1" }

	filter "*"

-- Third party dependencies
include "third_party" 

-- GLSLC helpers
dofile( "util/glslc.lua" )

-- Projects

project "main"
	local sources = { 
		"main/**.cpp",
		"main/**.hpp",
		"main/**.hxx"
	}

	kind "ConsoleApp"
	location "main"

	files( sources )

	dependson "shaders"

	links "utils"
	links "x-volk"
	links "x-stb"
	links "x-glfw"
	links "x-vma"

	dependson "x-glm"

project "shaders"
	local shaders = { 
		"main/shaders/*.vert",
		"main/shaders/*.frag",
		"main/shaders/*.comp",
		"main/shaders/*.geom",
		"main/shaders/*.tesc",
		"main/shaders/*.tese"
	}

	kind "Utility"
	location "main/shaders"

	files( shaders )

	handle_glsl_files( glslcOptions, "assets/main/shaders", {} )

project "main-bake"
	local sources = { 
		"main-bake/**.cpp",
		"main-bake/**.hpp",
		"main-bake/**.hxx"
	}

	kind "ConsoleApp"
	location "main-bake"

	files( sources )

	links "utils"
	links "x-tgen"
	links "x-zstd"

	dependson "x-glm" 
	dependson "x-rapidobj"

project "utils"
	local sources = { 
		"utils/**.cpp",
		"utils/**.hpp",
		"utils/**.hxx"
	}

	kind "StaticLib"
	location "utils"

	files( sources )

project()

--EOF
