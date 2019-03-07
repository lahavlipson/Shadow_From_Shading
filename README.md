# Shadow_From_Shading

Example of how to run:
`sudo python3 experiment.py --niter 10 --ep_len 100 --batch_size 8  --workers 8 --cuda`



TODO:
- Have network generate just the shadows and then overlay that on the input.
- Have experiment.py write self.train_losses to a graph each epoch, so we can track progress.
- Have experiment.py write the learned model weights to a file each epoch, so we can stop the training and pick up where we left off. We need to make sure to delete old network weights since each file is like 1.2Gb.

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
