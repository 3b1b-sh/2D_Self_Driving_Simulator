## HMM Agent

To driving manually, run the following command:

```bash
python cs181pj-car-hmm/drive.py -l lombard -i none
```

- You can steer by either using the arrow keys or 'w', 'a', and 'd'. The up key and 'w' accelerates your car forward, the left key and 'a' turns the steering wheel to the left, and the right key and 'd' turns the steering wheel to the right. Note that you cannot reverse the car or turn in place. 
- In this phase, no inference has been implemented, so you are unable to see any of the other cars. It is hard to complete the game now.

---

The transfer learned transfer probabilities have already been stored in the `./cs181pj-car-hmm/learned`

(Optional) Before driving with inference, transfer probabilities need to be learned by running the following command:

```bash
python cs181pj-car-hmm/learn.py -a -d -k 3 -l lombard
```

To drive with exact inference, run the following command:

```bash
python cs181pj-car-hmm/drive.py -a -d -k 3 -l lombard -i exactInference
```

To drive with particle filter, run the following command:

```bash
python cs181pj-car-hmm/drive.py -a -d -k 3 -l lombard -i particleFilter
```

- `-l` specifies the map, with available choices including `lombard`, `small`, and `small2`
- `-a` specifies to use the agent which follows a fixed path and stops when it detects other cars nearby. After the simulation starts, the agent will remain stationary until it has gathered enough observational data for the HMM to estimate the positions of other vehicles.
- `-d` specifies to show other cars in the GUI. It cannot be utilized by the agent.
- `-k 3` specifies the number of other cars;

## Approximate Q learning

To run approximate Q learning, run the following command:

```bash
python cs181pj-car-q/qdrive.py -k 3 -l small2 -g -i 50 -t
```

- `-l small2` specifies the map, and small2 is recommended for approximate Q learning;
- `-g` specifies to show the GUI in the first several iterations;
- `-i 50` specifies the number of iterations;

When training is done, it will automatically start trial runs and calculate the rate of success.

Due to the time-consuming nature of training and its inherent randomness, we provide a set of pre-trained weights.
To run approximate Q learning agent with pre-trained weights, remove the `-t` flag, like this:

```bash
python cs181pj-car-q/qdrive.py -k 3 -l small2 -g
```
