# How i want the demo logic to work:

**DO NOT MOVE TO NEXT STEP WITH OUT CLICK OF A BUTTON IN THE GUI AS IT WOULD GIVE ME TIME TO EXPLAIN**

1. User uploads 4 images (1 for each direction) and a audio file (for emergency vehicle detection)
2. User Click start demo
3. images are passed to the model orchestrator
4. Model orchestrator runs the model in the following order:
    - (Model 1)Traffic detection - returns a dict with count of vehicles in each direction
    - (Model 2)Emergency vehicle detection - returns a dict with count of emergency vehicles in each direction
    - (Model 3)Siren detector - returns a boolean indicating if a siren is detected in the audio file
5. if the lane has emergnecy vehical and siren is detected, then the traffic light for that lane will turn green for 30 seconds and the other lanes will turn red (in case a lane has a ongoing green light, and ambulance apper in other lane, then give a 3s buffer to this lane and then make a switch)
6. if no emergency vehicle is detected, then the traffic light will operate in a normal cycle
7. a infrence of (Model 4)traffic density predictor will run and we would smooth out the output of Model 1 with the help of Model 4 . this would give us a updated dict
8. Now this dict is passed to the DQN model which will give us the optimal action to take (which lane to turn green)
9. then the gui is updated and the traffic light is changed according to the action given by the DQN model
10. in animation vehical would move 1-2/s from a lane. after that lane signal turn green run another infrence of DQN model and update the traffic light accordingly