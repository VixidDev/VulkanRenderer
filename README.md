# Vulkan Renderer

University project to create a Vulkan renderer of the [Sun Temple](https://developer.nvidia.com/ue4-sun-temple) scene from [NVIDIA ORCA](https://developer.nvidia.com/orca). 

Implements a standard forward rendering pipeline with shadow mapping with a single light source, aswell as a deferred rendering pipeline with lights at each of the braziers in the Sun Temple. Has debug visualisations for visualising the current mipmap used per fragment, linearised depth, partial derivatives of per-fragment depth, overdraw and overshading.

Also implements a post-processing effect to add a mosaic effect ontop of the forward rendering pipeline.

https://github.com/user-attachments/assets/0bc2aa02-387c-45d7-a9fb-4fbd3f15f078

### Debug Visuals

| Debug Setting | Image |
|--- |--- |
| Mipmap level | <img width="840" alt="mipmap_levels" src="https://github.com/user-attachments/assets/5657373c-31d0-4e20-aa0e-0a80baf8d747" /> |
| Linearised fragment depth | <img width="840" alt="linearised_depth" src="https://github.com/user-attachments/assets/ed82bd9a-aaa7-4692-bcff-879a9f02a1ef" />  |
| Partial derivatives of per-fragment depth | <img width="840" alt="per_fragment_depth_derivatives" src="https://github.com/user-attachments/assets/237cf0c0-5a36-428c-bc5b-67a2424f56a7" /> |
| Overdraw | <img width="840" alt="overdraw" src="https://github.com/user-attachments/assets/d0cea3c4-3a8e-43d2-9b56-bc609ae1caf0" /> |
| Overshading | <img width="840" alt="overshading" src="https://github.com/user-attachments/assets/2b7bc7d4-055e-4ab4-a7df-94adbdb74b85" /> |

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
