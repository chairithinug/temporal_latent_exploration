<!-- markdownlint-disable MD013 -->
## Reference
- [Predicting Physics in Mesh-reduced Space with Temporal Attention](https://arxiv.org/abs/2201.09113)
## Dataset and Lib
- [PhysicsNemo](https://github.com/NVIDIA/physicsnemo)
## Mesh Face
mat_delaunay_filtered.npy
## Example of a training trajectory
![Training trajectory 0](small_animation/0_xyz-2.gif)
## Reconstruction (test trajectory 90) from GAE with a triplet loss
![Reconstruction w/ triplet](small_animation/mesh_animation_grid-2.gif)
## Latent evolution of test trajectory 90 from GAE with a triplet loss
![Reconstruction w/ triplet](small_animation/tracking_animation_triplet.gif)
## Interpolation (test trajectory 90) from GAE with a triplet loss
![interpolate](small_animation/test_interpolate_optimized-2.gif)
## Granular walk of test trajectory 90 between timestep 121 and 122, broken down into 11 steps (endpoints included)
![granular](small_animation/granular90.gif)
## Coarse walk of test trajectory 90 between timestep 0 and 400, broken down into 21 steps (endpoints included)
![coarse](small_animation/coarse90.gif)
## Extrapolation (test trajectory 90) from GAE with a triplet loss
![extrapolate](small_animation/test_extrapolate_optimized-2.gif)

## Scripts for SLURM
Under scripts/
## Used code
Under main_code/
## Weights and Checkpoints
Under checkpoints/
no model_l2_256hidden_best.pt due to large space required
## Env
See requirements.txt
