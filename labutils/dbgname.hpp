#ifndef DBGNAME_HPP_741344AB_2708_4F93_90A1_FE232523098A
#define DBGNAME_HPP_741344AB_2708_4F93_90A1_FE232523098A

// Config: C5_ENABLE_DEBUG_NAMES
//
// Some pre-processor gymnastics follow to set the C5_ENABLE_DEBUG_NAMES flag.
// It will control if debug names should be set. By default, it will depend on
// the NDEBUG macro, which was used previously to enable/disable the validation
// layers and the VK_EXT_debug_utils extension. It is possible to control the
// setting by defining COMP5822MCFG_ENABLE_DEBUG_NAMES (e.g., from Premake or
// similar).
//
// I use the C5_ prefix here; COMP5822M_ ended up being too long...
#if !defined(COMP5822MCFG_ENABLE_DEBUG_NAMES)
#	if !defined(NDEBUG)
#		define C5_ENABLE_DEBUG_NAMES 1
#	else
#		define C5_ENABLE_DEBUG_NAMES 0
#	endif
#else
#	define C5_ENABLE_DEBUG_NAMES COMP5822MCFG_ENABLE_DEBUG_NAMES
#endif

#if C5_ENABLE_DEBUG_NAMES
#	include <cstdio>
#	include <source_location>
#endif

// C5_DBGNAME_ARG()
#if C5_ENABLE_DEBUG_NAMES
#	define C5_DBGNAME_ARG(x) , x
#else
#	define C5_DBGNAME_ARG(x)
#endif

// C5_DBGNAME_{DECL,DEFN}() 
#define C5_DBGNAME_DECL()                                           \
	C5_DBGNAME_ARG(char const* = nullptr)                           \
	C5_DBGNAME_ARG(std::source_location const = std::source_location::current())  \
	/*ENDM*/
#define C5_DBGNAME_DEFN()                                           \
	C5_DBGNAME_ARG(char const* aDebugName)                          \
	C5_DBGNAME_ARG(std::source_location const aDbgSrcLoc)           \
	/*ENDM*/

// C5_DBGNAME_SET_NAME()
#if C5_ENABLE_DEBUG_NAMES
#	define C5_DBGNAME_SET_NAME(dev,x,type) do {                     \
		char nameBuff[256]{};                                       \
		if( aDebugName ) std::snprintf( nameBuff, 255, "%s", aDebugName );  \
		else std::snprintf( nameBuff, 255, "%s:%d", aDbgSrcLoc.file_name(), aDbgSrcLoc.line() );  \
		VkDebugUtilsObjectNameInfoEXT ninfo{};                      \
		ninfo.sType         = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;  \
		ninfo.objectType    = type;                                 \
		ninfo.objectHandle  = reinterpret_cast<std::uint64_t>(x);   \
		ninfo.pObjectName  = nameBuff;                              \
		                                                            \
		if( auto const res = vkSetDebugUtilsObjectNameEXT( dev, &ninfo ); VK_SUCCESS != res ) {  \
			std::fprintf( stderr, "Note: vkSetDebugUtilsObjectNameEXT() failed for %s\n", aDebugName );  \
		}                                                           \
	} while(0) /*ENDM*/
#else
#	define C5_DBGNAME_SET_NAME(dev,x,type) do {} while(0)
#endif

#endif // DBGNAME_HPP_741344AB_2708_4F93_90A1_FE232523098A
