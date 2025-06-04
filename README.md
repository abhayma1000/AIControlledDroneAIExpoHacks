# AI drone control

This is not our core submission, but something we tried. We believe this is promising, but didn't have enough time to fully flush it out


# How it works

* Taking in the controller controls: (left x, left y, right x, right y) and the velocities and angles of the previous time step, predict the next time step velocities and angles
* ```create_drone_model.ipynb``` is the file to create the machine model and ```drone_physics.py``` runs the model along with a standard drone to compare. ```original_drone.py``` is just a simple drone physics


# Why
* It is difficult to model the exact dynamics of the drone. Instead of relying on complicated physics with too many parameters to get correct, use a data based approach
* 

# How we got the data
* Instead of using the Tesseract Nano, which there is limited telemetry data for, we use a DJI mavic
* https://huggingface.co/datasets/nominal-io/dji-drone-telemetry/tree/main/Palos%20Verdes

# Future
* In the future, train this model using data generated from the Tesseract Nano instead.
* We believe that accurate data based modelling is better than any realisticly achievable physics based modelling that would require expensive equipment and a lot of time
* If we had more time, would investigate Offline Reinforcement Learning to train this model

# Get started
* pip install -r requirements.txt
* Run files