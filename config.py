# dir with ESC50 data
esc50_path = 'data/esc50'

runs_path = 'results'
# sub-epoch (batch-level) progress bar display
disable_bat_pbar = False
#disable_bat_pbar = True

# do not change this block
n_classes = 50
folds = 5
test_folds = [1, 2, 3, 4, 5]
# use only first fold for internal testing
#test_folds = [1]

# sampling rate for waves
sr = 44100
n_mels = 128
hop_length = 512
#n_mfcc = 42
n_fft=1024
# The number of time‐bins in the final spectrograms.
# With sr=44100, hop_length=512, 5 seconds audio ≈ floor((44100*5)/512) = 430 frames; round up to 431 if padded to full.
time_frames = 431

model_constructor = "AudioResNet12(n_mels=config.n_mels, time_frames=config.time_frames, num_classes=config.n_classes)"


# ###TRAINING
# ratio to split off from training data
val_size = .2  # could be changed
device_id = 0
batch_size = 64
# in Colab to avoid Warning
num_workers = 2
# for local Windows or Linux machine
# num_workers = 6#16
persistent_workers = True
pin_memory=True
epochs = 80 # model + optimizer combo apparently does not need that many epochs, usually converges earlier
#epochs = 1
# early stopping after epochs with no improvement
patience = 10 # i have no patience, so far most have stopped improving around 60-70
lr = 3e-4
weight_decay = 2e-3
warm_epochs = 10
gamma = 0.9 # more gentle decay
step_size = 5

# ### TESTING
# model checkpoints loaded for testing
test_checkpoints = ['terminal.pt']  # ['terminal.pt', 'best_val_loss.pt']
# experiment folder used for testing (result from cross validation training)
test_experiment = 'results/2025-06-01-18-47'
#test_experiment = 'results/sample-run'