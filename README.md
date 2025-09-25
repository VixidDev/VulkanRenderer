# Vulkan Renderer

Initially a university project to create a Vulkan renderer of the [Sun Temple](https://developer.nvidia.com/ue4-sun-temple) scene from [NVIDIA ORCA](https://developer.nvidia.com/orca). 

Implements a standard forward and deferred rendering pipeline with shadow mapping support for directional lighting and omnidirectional shadow mapping for point lights. Employs a optimisation method to pack the entire tangent-bitangent-normal (TBN) matrix into 1 vertex attribute of format `VK_FORMAT_A2R10G10B10_UNORM_PACK32` and then decode the matrix in the vertex shader and later use it to sample from a normal map (original normals are also passed to the shader as a fallback in case of input models with degenerate UV mapping that causes the TBN matrix to have NaNs). Uses a custom file format to define light position and properties. Uses ImGUI to provide a settings and debug interface for various debug options and visualisations such as:
- Normals
- Mipmap level of current fragment
- Linear depth
- Partial derivative of fragment depth
- Overdraw
- Overshading
- PBR Distribution function
- PBR Geometry function
- PBR Fresnel function
- Inspect each shadow map, including each face of cubemapped shadow maps
- Inspect the view of the sun direction
- Adjust depth biases and emissive strength
- Adjust colour and intensity of each light in the scene

Also implements a modular post-processing effect system to add any number of effects post-scene rendering. Currently has just a mosaic post-processing effect.

| Debug Setting | Image |
|---|---|
| Normals | |
| Mipmap level | |
| Linear depth | |
| Partial derivatives | |
| Overdraw | |
| Overshading | |

## Usage

<b>Dependencies</b>
- premake5
- Visual Studio 2022 (or later)

To compile on Windows:

1. Git clone the repository
2. Run `premake5 vs2022`

`main-bake` must be ran first to bake the custom files and binaries used for loading. Set `main-bake` as startup project in Visual Studio and run in Release mode, do not run in debug mode, it is significantly slower and since it uses Zstd it heavily benefits from compiler optimisations.

After `main-bake` has completed set `main` as startup project and run as normal.