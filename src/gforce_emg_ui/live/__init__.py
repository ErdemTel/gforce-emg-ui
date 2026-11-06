"""Live Classifier: frozen emg-classifier predictions from the live DataHub EMG stream.

Qt-free. emg_classifier (and scikit-learn) are only imported when a model is loaded,
so the recording app works without them. Nothing here changes recordings or
trains/adapts the model. Optional robot output (robot_out): only while the operator has
armed it, the shown s goes to a robot bridge over UDP; the bridge decides what the robot
does with it (an invalid s is sent as null, never 0).
"""
