# Julia_exercise_projects
Number of exercises in Julia.

## Quasi-general raycasting in General Relatvity

-Home built, high-performace GR raycaster and visualizer, centered around the ADM formalism of General Relativity

-For the relevancy to this project, the formalism by Richard Arnowitt, Stanley Deser and Charles W. Misner 'decouples' the timelike coordinate from the rest of the spacelike coordinates, thus reducing the number of euqtions of motion from 8 to 6, by eliminating the geodesic equation of the temporal coordinate. 

-Entirely coordinate-indepent: only the description of (spacelike) infinity is required to render an image. Visualization of timelike paths/images is also supported.

-Main `struct` is entirely symbolic, and pre-computes the symbolic coordinate evolution equations. Bencharmking suggests that is this is roughly 10% improved over the standard, optimized way of integrating the EOMs.

-Future improvements could include RGB-translated redshift (which is computed, but not used at the moment) or a metric-independent, four-momenta based ray termination algorithm.

-Tested metrics include the Schwarschild, the Kerr-, Minkowski and Alcubierre (sublight) metrics.

Example renders:

### Schwarschild blackhole between two checker-texture plates, located at the z-like horizon

![tracker_texture_SCH_blackhole](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/f3a1fb00-ccdc-4916-be72-3ae92f2a87d2)

### Schwarschild blackhole in low orbit around an imaginary planetary system with rings (note the Einstein ring)

![ADM_Imaginary_LEO](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/adbb68be-ef65-49ed-87d4-9d558967faac)

### Kerr Blackhole inside the same imaginary low-orbit texture as the Schwarschild one (a = 0.95)
![ADM_KERR_high_res](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/57496b18-d90f-47f8-818f-be4619b9bf1f)

### Kerr blackhole inside same checkerboard texture (a = 0.95)

![ADM_KERR_tracker](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/93f5792d-22de-42c1-95ff-713e59157573)

### Alcubierre metric (position chosen such that null geodesics in flat space would reach the "buble" at the time of taking the photo)

![ALC_tracker_test_HQ](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/71d61bce-074a-49f1-bca3-49963f17b26f)


## Machine learning - analyzing the MNIST dataset using home-built `neural_network struct`.

-Home-built, medium-performance feed-forward neural network implementation using base Julia library

-3 layer network capable of 96% accuracy on the test-split of the MNIST dataset. 

-main `struct` capable of constructing any feed-forward network architecture

-Potential future improvements include implementing other gradient-descent methods such as ADAM, and type declaration in main `struct`, or symbolic differention using `Symbolics.jl`, to achieve high performance

![ML_training performance](https://github.com/ArchHem/Julia_exercise_projects/assets/84734676/b84aff0b-d320-4706-90c8-5ed066b47079)

