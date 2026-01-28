Objective: Map Human Vowel Perception across the entire (physiologically possible)
vowel space and compare with Artificial Neural Network vowel classification

vowel_grid.py generates a list of formant 2D coordinates called inside_points.csv

vowel_stims contains a list of vowels (WAV files) synthesized artificially from those coordinates

run_experiment.py contains the experiment which used vowel_stims and yielded behavioural_data

group_analyses analyses the behavioural data and produces the human perceptual map

neural_net_vowel.py uses Training_data_participants.csv to classify vowels and produces another perceptual map

NeuralNet_2DAnimation.mp4 visualises the affine linear transformations and point wise nonlinear transformations that neural_net_vowel.py uses to classify vowels from the training data.
