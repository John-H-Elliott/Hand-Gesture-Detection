# Structure
### Structure: Hardware
Code must have all modules and Python installed to be able to run correctly. Also, a camera is required to run this application.

If running ```home_assiastant.py``` without mocking it a home assiastant server needs to be set up with a WIFI-connected bulb regesitered. Then a long-term key and the lights ID should be used in a ```.env``` file using the names  *HOME_ASSISTANT_LONG_TERM_KEY* and the _LIGHTING_ID_, respectively.

### Structure: Files
To run this application ```main.py```,```home_assiastant.py```,```gesture_recognizer.task``` must all be in the same directory. Also, a ```.env``` file needs to be included which provides the *HOME_ASSISTANT_LONG_TERM_KEY* and the *LIGHTING_ID* (Not required to be valid if setting mock = True).

### Structure: Code
```main.py``` is the main application. It first detects the hand landmark key points using the ```gesture_recognizer.task``` bundled model with Mediapipe and then detects the gesture by either another Mediapipe model in ```gesture_recognizer.task``` or by using the key points to calculate a set of vectors which can be used to determine a likely gesture.

This application takes two inputs *calculated* and *mock*. *Calculated* determines which method for detecting gestures is used while *mock* determines if ```home_assiastant.py``` should ignore sending a Post request and just mock this interaction instead.

```home_assistant.py```
This application handles the Post request that is sent to the Home Assistant server. This is done through the light_service() function. It will mock this if *mock* is set to True. It also takes *state* which is used to determine the URL address the Post request is made to and *brightness* which is used if the *state* is set to **turn_on**.

If run alone it will run the test_light_services() function which simply tests the Post request for each state.

# Run Instructions
To run the application the file structure and hardware above need to be present and correct. Then run the command:
```
 python3 main.py
```

This runs detection(calculated, mock). To change the values of these inputs simply change them (True or False) in ```main.py```.

This should start the application and utilize the connected camera to detect gestures. If connected correctly to a home assistant server it will send a Post requests allowing for control over the connected lighting.