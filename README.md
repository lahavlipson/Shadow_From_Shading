# Shadow_From_Shading

Example of how to run:
`python3 experiment.py --niter 10 --ep_len 28 --batch_size 7 --workers 1 --cuda`



TODO:
- Evolution class
    - Population = []
        - Will be the members (i.e. scene objects) of the population that are surviving. 
    - evolve()
        - Will 1) add new members to the population by mutating existing members, or 2) will add new members to the population by crossing over existing members. 
    - get_data()
        - Will call generate_images() on all the scene objects in the population, concatenate them, and return the two tensors as (inputs,labels)
    - purge(evaluation scores)
        - Will remove all but the top ~50 members of the population, ranked by how well/poorly the neural network did on them
- Dataset class 
    - A pytorch dataset class that interacts with an evolution class to generate data for the neural network.
