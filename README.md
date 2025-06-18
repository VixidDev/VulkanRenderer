# Vulkan Renderer

University project to create a Vulkan renderer of the [Sun Temple](https://developer.nvidia.com/ue4-sun-temple) scene from [NVIDIA ORCA](https://developer.nvidia.com/orca). 

Implements a standard forward rendering pipeline with shadow mapping with a single light source, aswell as a deferred rendering pipeline with lights at each of the braziers in the Sun Temple. Has debug visualisations for visualising the current mipmap used per fragment, linearised depth, partial derivatives of per-fragment depth, overdraw and overshading.

Also implements a post-processing effect to add a mosaic effect ontop of the forward rendering pipeline.

#### Keybinds

Right-click the screen after launching to focus mouse in the application so you can move around with WASD.

- `1` - Forward rendering pipeline
- `2` - Mipmap level visualisation
- `3` - Linearised fragment depth
- `4` - Partial derivatives of per-fragment depth
- `5` - Moasic post-processing effect (Only works with with forward rendering)
- `6` - Overdraw visualisation
- `7` - Overshading visualisation
- `8` - Deferred shading pipeline

## Usage

To compile on Windows:

```bash
premake5 vs2022
```

`main-bake` must be ran first to bake the custom files and binaries used for loading. Set `main-bake` as startup project in Visual Studio and run in Release mode, do not run in debug mode, it is significantly slower and since it uses Zstd it heavily benefits from compiler optimisations.

After `main-bake` has completed set `main` as startup project and run as normal.