"""Live Classifier: frozen emg-classifier predictions from the live DataHub EMG stream.

Qt-free. emg_classifier (and scikit-learn) are only imported when a model is loaded,
so the recording app works without them. Display-only: nothing here controls a robot,
changes recordings, or trains/adapts the model.
"""
