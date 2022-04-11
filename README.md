# Gesture Detection Neural Network 

## Content
1. [Installation](#setup)
2. [Neural Network](#neural-network)
   1. [Preprocessing](#preprocessing)
   3. [Training](#training)
   4. [Using the Network](#neural-network-package)
3. [App Engine](#app-engine)
4. [Network Modes](#network-modes)
   1. [Test Mode](#test-mode)
   2. [Live Mode](#live-mode)
      1. [Mandatory Gestures Slideshow](#mandatory)
      2. [Optional Gestures Slideshow](#optional)
      3. [Tetris](#tetris)
5. [Remaining Folders](#remaining-folders)
   1. ['Data' Folder](#data)
   2. ['Tests' Folder](#tests)


# Setup
## Requirements
- Python >= 3.6
- Camera connected to your computer/laptop
- possibility to run python multiprocessing (cpu_count >= 4, will be explained in an upcomming chapter about [Game Engine](#app-engine)


## Installation
Open a terminal, cd into the root folder of your project and type:
```shell
pipenv install
```
or install the requirements listed in 'Pipfile', which lays in the project's root directory


## Start the App
The system is now setup and you are ready to run the applications.


There are three different live applications.

- [Slideshow (mandatory gestures)](#mandatory)
- [Slideshow (all/optional gestures)](#optional)
- [Tetris](#tetris)

Check in the chapters of each application how to start and use it.

# Neural Network
## Preprocessing

For preprocessing, there is a file 'data_preprocessor.py' in the folder 'neural_net'.
It contains a class, with a few parameters for preprocessing.

### Included Frames

The parameter including_frames, tells the preprocessor how many frames from the camera to combine into one input row for the network
If set to 20, which is the default, The first frame will be combined with the next 20 frames. For this, 
the difference of the selected elements (which are stated in line 18 and 19 in the class) will be calculated relatively to previous frame.
X and Y coordinates are taking care of separately for all the features
Additionally, the distances from the thumb to the mouth on the side the hand are calculated for all the included frames.

After the first frame is combined with the upcoming ones, the second frame will be combined in the same way with the following frames.
Thus, the number of rows after the preprocessing can be calculated by subtracting the included frames from the total number of inputs before the preprocessing.
Each frame occurs in multiple rows after preprocessing, but everytime on different places in the row. One time it is on first place, another time in the middle.
How often a frame occurs in separate preprocessed rows depends on the variable included_frames.


### Relative to first Frame

With the parameter relative_to_first_frame, the data will be calculated relative to the first of the included frames, instead of to the previous one.
Following table will represent inputs before preprocessing

| left_index_x |
|-------------:|
|          0.0 |
|          0.1 |
|          0.2 |
|          0.3 |

Setting included_frames = 3 would result into 3 columns for one preprocessed row like shown in following table. 

| Relative mode              |    Column 1     |    Column 2     |     Column 3     |
|:---------------------------|:---------------:|:---------------:|:----------------:|
| Relative to previous frame | 0.1 - 0.0 = 0.1 | 0.2 - 0.1 = 0.1 | 0.3 - 0.2 = 0.1  |
| Relative to first frame    | 0.1 - 0.0 = 0.1 | 0.2 - 0.0 = 0.2 | 0.3 - 0.0 = 0.3  |


### Percentage Majority

The last parameter to select it the percentage_majority, which is set to 50% by default.
Since multiple frames are combined into one row, there are two different labels in the ground truth column for many cases.
By changing the parameter, more or less labels of the gesture are required, so this label will be used as the final ground truth.

If there are 20 frames, 12 of them are set to 'rotate' and the remaining 8 to 'idle', the default case would choose 'rotate', because it occurs more often.
If the percentage_majority was set to 0.75 (75%), the ground truth would be 'idle', because 12 / 20 = 0.6, which is less than 0.75


### Body Size Scaler

The final parameter in the constructor is a class, which scale the data to the body size, in order to be able to compare 
someone standing right in front of the camera to someone standing in the back performing the same gesture
The class lives in another file 'data_scaler.py' in the same directoryl
Passing the instance of this class to the constructor was only done because of unit tests, since the use of Dependency Injection can be very helpful for such tests. 
More about that in the part about [unit tests](#tests)

The scaler will take the x and y coordinates of the left and right shoulder for all the included frames.
For each of the frames, the Pythagoras will be calculated, to get the 2-dimensional distance of the two shoulders.
This distance is similar to the distance of a swipe gesture or of the diameter of the imagined circle when performing a rotation 
and thus, it is suitable to scale the values.

The result of the scaler will be a vector with the size of the included frames, because every frame needs to be scaled with a different value, 
to cover a person moving towards or away from the camera during the time of the included frames.
This vector will not be calculated normally, but every element of the vector will be the factor, the same row in the matrix will be multiplied to.
Following example will make it easier to understand:

         | 1 | 4 | 3 |        | 2 |          | 1 * 2 | 4 * 2 | 3 * 2 |          
     A = | 2 | 1 | 5 |    v = | 1 |   =>     | 2 * 1 | 1 * 1 | 5 * 1 | 
         | 1 | 1 | 2 |        | 3 |          | 1 * 3 | 1 * 3 | 2 * 3 |


### Used features

The features used in preprocessing next to the distances from thumb to mouth are the follwoing:

"left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
"left_elbow", "right_elbow", "left_shoulder", "right_shoulder"

X- and Y-coordinates are used for all of them.


## Training

The file 'gesture_detection.py' also in the folder 'neural_net' was used to train the model to detect gestures.
The script in the main function was used to preprocess different csv files containing the preprocessed data which will be required for training combined
These files where stored in the sub-folder 'preprocessed_data' in different variants.

To train the model, another function in the same file was called with the data from the just saved files containing the preprocessed data.
The function contained parameters and was always changed to use different configurations for the network, 
like a different shape, activation function, optimizer and so on.
At the end of a training the session was saved in the folder 'saved_configs' grouped by included features and some of the parameters, so it can be used again for the applications.

Next to the mandatory saving, which is a function of the package, we created another save function to also store additional data during and after training
This allowed us to continue to train on a config, where we did not use enough iterations in the first place


## Neural Network Package

As already described, the Code to run the network, including all the necessary functionality,  
like PCA, Encoder etc. lives in a submodule, in the folder 'nn_framework_package'

Everything related to that code is described in the README.md of this repository


# App Engine

The App Engine is a class inside 'app_engine.py' in the 'app_engine' folder. It is starting the video capturer, 
reading and processing the images and feeding them into the network, so it can predict on them. The class will be instantiated as a background task in the sanic server,
so it will boot up, when starting the server. The main function, contains the endless loop, which will start separate processes and pipes.
The processes are created as child processes of the main loop, so none of them needs to be waited on and they will be terminated and closed when the main process will stop.

The processes are as follows:

- image_to_frame_process (reads image from the camera, shows the video in a window and converts the results of the pose into a row)
- network_preprocessing_process (reading the rows from previous process and preprocessing it and applying the PCA depending on the config passed to the constructor. This process will also sample the frames up to 30 FPS, in case the computer or the camera will provide less)
- network_prediction_process (scaling the data with the used scaler in the network and predicting. Every prediction will be sent through the next pipe including their confidence)
- action_process (reading all the predictions and confidences, after having the confidence of more than 1.8 which takes at least two confident predictions, it will send an action to the next pipe. No gesture can be performed two times in a row except idle. Idle will reset the confidence and also allow to send the same gesture again)

After starting all the processes, the main function is ready to use, which can be seen by a console log.
The engine will show the video with the poses being recognized, but not send the results to the network.

As soon as the user opens the browser, the emit_events function will be called. This will toggle the boolean 'should_run', to start accepting gestures and start predicting.
The main loop will notify the process via another pipe and the process will work at the same time only communicating through Pipes.

The emit_events function will constantly check, if the action pipe contains data, and if yes the websocket will send the string coming from that pipe.
A function is passed to the constructor of the AppEngine, to define how to convert prediction labels to actions.
It will usually consist of a few if statements, returning something based on the prediction like, if the prediction was 'flip_table', return 'up'.

The class takes a few parameters that do not need to be changed because they are set in the separate applications.
These parameters allow the class to be reused for all the applications.



# Network Modes
## Test Mode

The Test Mode lives in the folder 'test_mode' in the file 'test_mode.py'. This file is run as a script and accepts two arguments.
The Name of the input csv file and the name of the output csv file. Both parameters are optional and set to default values 'video_frames_input.csv' and 'performance_results.csv'.
Change the default values can be done by running it in the terminal like in the following example:

```shell
python test_mode.py --input_csv="my_input.csv" --output_csv="my_output.csv"
```

The input is a csv file which was converted from mp4 using mediapipe. 
The output will be a csv file containing only indices, timestamp and the predicted event at that timestamp or idle.

## Live Mode

The camera index for the applications is set to 0 by default, but can be added as an argument to any of the app-scripts. 
In order to change to a different setting, simply type following command in the terminal to start the app with different settings:
```commandline
python app.py --camera_index=1
```
```commandline
python app_optional.py --camera_index=1
```
```commandline
python tetris_app.py --camera_index=1
```

### Mandatory

The application with mandatory gestures lives in 'app.py' in the folder 'slideshow'.
It starts a sanic server and adds the [app engine](#app-engine) as a background task as previously described.
It renders html, css and javascript from the 'static' folder, where a websocket will be created. The static folder contains the content of the slideshow, which will be the content of the application 
The sanic server will connect to the websocket and start the prediction as already described in the app engine.

The function 'handle_mandatory_gestures' is defining what action to send depending on the received prediction label.

The application will be available on 'http://0.0.0.0:8000'

### Optional

Same as the application with mandatory gestures, the one with optional gestures also starts a sanic server including the app engine in the same way.
The file is called 'app_optional.py' and lives direclty next to the mandatory one. It renders the same html, css and javascript and also starts the server on port 8000.

Thus, the application will also be available on 'http://0.0.0.0:8000', but can not be run at the same time as the app with mandatory gestures.

Inside there is another function handling all the gestures (mandatory and optionals).

### Tetris

The last live mode is starting the game Tetris. The app lives in the folder 'tetris' in the file 'tetris_app.py'.
Same as the other two live modes it starts a sanic server and the app engine as a background task. 
It also has a function to convert prediction labels to actions.

The only difference is the content of the static folder which will be rendered and the port of the server.
The static folder now contains the content to play a tetris game.

The server is available on 'http://0.0.0.0:8001'.

Running the server at the same time as one of the other applications would be possible, but is not recommended, 
since it will probably decrease the performance of all applications. 


# Remaining Folders

## Data

This folder contains the csv file converted with mediapipe from mp4 videos. Every data used for training lives here.
The script inside 'combine_data.py' was used to combine mediapipe_frames and elan_annotations. 
The validation videos from WueCampus also live in the output_csv folder, but were not used for training, so they could be used for validation of the performance score.

## Performance Score
The performance score contains the scripts from the final-getting-started repo, to check the performance of our neural network configurations.


## Plots

In this folder, some plots that were used in the presentation were saved after training was done.


## Presentation

This folder contains our presentation and a script used for visualizations in the presentation.
Additionally, there are a few more screenshots and plots used for comparison in the presentation.

### Teaser Video

In the presentation folder, there is also a file called 'ML Teaser.mp4'. This is our teaser video of our project

## Tests

This folder contains unit tests for some of the functionality of our project.
There are tests for the following classes or functions:

- DataPreprocessor
- DataScaler (body size scaler called in the preprocessor)
- AppEngine.action_from_prediction (algorithm to choose an action after a confidence was reached)
- OneHotEncoder (although the class lives in the package in another repository now, the tests stayed there, so they can be run all together)

The file inside the same folder 'test_processing.csv' was just some dummy data used for testing the preprocessing