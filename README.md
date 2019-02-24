# Shadow_From_Shading

### Here is a list of objectives, with general specifications and rough due dates.

~~cross out elements once completed~~

- Shape class (Aim for Feb 26th before midnight)
    - Rotate method (pitch and yaw)
    - Scale method (axis=(1||2||3), degree)
- Scene class (Aim for March 2nd before midnight)
    - Note: This scene will have an invariant background. Maybe something like the cornell box
    - generate_images()
        - Will return 2 images, shadowed and unshadowed
    - Crossover(Scene other_scene)
        - Return a new scene made with two random subsets of the current scene and the scene passed as an argument
    - Mutate method()
        - Will call one of the following methods to modify the scene
    - Scale object
    - Rotate object
    - Add object
- Evolution class (Aim for March 4th before midnight)
    - Population = []
        - Will be the members (i.e. scene objects) of the population that are surviving. 
    - evolve()
        - Will 1) add new members to the population by mutating existing members, or 2) will add new members to the population by crossing over existing members. 
    - get_data()
        - Will call generate_images() on all the scene objects in the population, concatenate them, and return the two tensors as (inputs,labels)
    - purge(evaluation scores)
        - Will remove all but the top ~50 members of the population, ranked by how well/poorly the neural network did on them
- Dataset class (Aim for March 9th before midnight)
    - A pytorch dataset class that interacts with an evolution class to generate data for the neural network
- CNN class (Aim for March 10th before midnight)
    - Fully convolutional autoencoder. I can code this up in 20 minutes, let’s do this part last.
- main.py (Aim for March 17th before midnight
    - Our main script
