# Autopilot Training Improvement Guide

## What Was Improved

### 1. **Enhanced Model Architecture (PilotNet-inspired)**

The new model is deeper and more sophisticated:

- **5 Convolutional layers** (vs 3 before) for better feature extraction
- **Batch normalization** after each conv layer for training stability
- **3 Dense layers** (vs 1 before) for better decision making
- **More parameters** → Better learning capacity
- **Dropout layers** (30%, 30%, 20%) to prevent overfitting

### 2. **Data Augmentation**

- **Horizontal flipping** with steering inversion (doubles your dataset)
- **Random brightness** adjustments (helps with lighting variations)
- Better generalization from limited training data

### 3. **Better Training Strategy**

- **Higher train/val split** (85/15 vs 80/20) - more training data
- **Learning rate scheduling** - automatically reduces LR when stuck
- **Early stopping** with more patience (10 epochs)
- **Model checkpointing** - saves best model automatically
- **Lower initial learning rate** (5e-4 vs 1e-3) for better convergence

### 4. **Improved Callbacks**

- TensorBoard logging for visualization
- CSV logging for analysis
- ReduceLROnPlateau for adaptive learning
- ModelCheckpoint for best model saving

### 5. **Better Autopilot Control**

- **Throttle scaling** - run at reduced speed for safety
- **Steering smoothing** - reduces jitter/oscillation
- **Better status display** - see what the model is doing
- **Emergency stop** - space bar for immediate stop
- **Auto-loads best model** - uses validation best, not final

## How to Get Better Results

### Step 1: Collect Quality Training Data

#### Tips for Good Data Collection:

1. **Drive smoothly** - No jerky movements
2. **Variety of scenarios**:
   - Straight sections
   - Left turns
   - Right turns
   - Different lighting conditions
   - Recovery from edge of track
3. **Balanced steering**:
   - Equal amounts of left/right turns
   - Not too much straight driving
4. **Consistent speed** - Don't vary throttle too much
5. **Minimum 500-1000 frames** - More is better!

#### Recording Process:

```bash
python autopilot_drive_train_picam2.py
# Choose option 1
# Press 'r' to start recording
# Drive for several laps
# Press 'r' to stop recording
# Press 'q' to quit
```

### Step 2: Train with New Settings

When training, you'll be asked:

1. **Use data augmentation?** → **YES** (recommended)
   - Doubles your dataset with flipped images
2. **Use lightweight model?** → **NO** (unless Pi is slow)
   - Full model is better but slower to train
3. **Number of epochs?** → **50-100**
   - More epochs = better learning (with early stopping)
4. **Batch size?** → **32** (default is good)
   - Larger batch = faster but needs more memory

### Step 3: Monitor Training

Watch for these signs:

#### Good Training:

- Val loss decreases steadily
- Train loss and val loss close together
- MAE below 0.15
- Early stopping triggers (means it found best point)

#### Bad Training (Overfitting):

- Val loss increases while train loss decreases
- Large gap between train and val loss
- High MAE on validation
- Training continues without improvement

**Solution**:

- Collect more data
- Enable data augmentation
- Reduce model complexity (use lightweight)

#### Bad Training (Underfitting):

- Both losses high and not decreasing
- MAE stays above 0.3
- Model predictions are random

**Solution**:

- Train longer (more epochs)
- Use full model (not lightweight)
- Check data quality

### Step 4: Test Autopilot Safely

```bash
python autopilot_drive_train_picam2.py
# Choose option 3
# Select your trained model
```

**Configuration:**

- **Throttle scale**: Start with 0.5 (half speed) for testing
- **Steering smoothing**: YES (reduces oscillation)

**Controls:**

- `a` = Autopilot mode (AI drives)
- `h` = Manual mode (you drive)
- `Space` = Emergency stop
- `q` = Quit

**Testing Process:**

1. Start at **0.5 throttle scale**
2. Let it run for 10 seconds
3. If stable, gradually increase to 0.7-0.8
4. Watch for:
   - Smooth steering (not zigzagging)
   - Consistent throttle
   - Staying on track

## Understanding Training Plots

After training, check `training_plot.png` in your session folder:

### Plot 1: Model Loss (MSE)

- **Train and Val should decrease together**
- **Final val_loss < 0.02** is excellent
- **Final val_loss < 0.05** is good
- **Final val_loss > 0.1** needs improvement

### Plot 2: Mean Absolute Error

- **MAE < 0.1** = Excellent predictions
- **MAE 0.1-0.2** = Good predictions
- **MAE > 0.3** = Poor predictions (needs more data)

### Plot 3: Learning Rate Schedule

- Should decrease over time (step-wise)
- Indicates when ReduceLROnPlateau kicked in

### Plot 4: Overfitting Check

- **Near zero** = Perfect fit
- **Positive** = Slight overfitting (okay if small)
- **Large positive** = Severe overfitting (bad)
- **Negative** = Model not learning training data

## Troubleshooting

### Problem: Car oscillates/zigzags

**Solutions:**

- Enable steering smoothing (already enabled by default)
- Reduce throttle scale (0.5-0.6)
- Collect smoother training data
- Train longer (let early stopping work)

### Problem: Car goes off track

**Solutions:**

- Collect more data including edge recovery
- Use data augmentation
- Train with more epochs (100+)
- Check if data has enough variety

### Problem: Val loss not improving

**Solutions:**

- Collect more data (need 500+ frames minimum)
- Enable data augmentation
- Check data quality (are labels correct?)
- Reduce learning rate (already set lower)

### Problem: Model too slow on Pi

**Solutions:**

- Use lightweight model
- Reduce image size (edit IMAGE_W, IMAGE_H)
- Use TensorFlow Lite (advanced)

## Advanced Tips

### 1. Data Balance

Check your `labels.csv`:

```bash
# Count steering distribution
cut -d',' -f2 labels.csv | sort -n | uniq -c
```

Should have roughly equal left/right turns.

### 2. Multiple Sessions

Train on multiple combined sessions:

```bash
# Collect data from different tracks/conditions
# Then manually combine sessions or use option 2 to select best
```

### 3. Fine-tuning

Load existing model and train more:

```python
# In train_model_on_session, add:
# model = _robust_load_model("previous_model.keras")
# Then continue training with new data
```

### 4. Hyperparameter Tuning

Edit these in code:

```python
# Learning rate (line ~330)
learning_rate=5e-4  # Try 1e-4 for slower, 1e-3 for faster

# Dropout rates (line ~325)
layers.Dropout(0.3)  # Try 0.2-0.5

# Patience for early stopping (line ~400)
patience=10  # Try 5-15
```

## Expected Performance

With **good training data** (1000+ frames, varied scenarios):

| Metric            | Excellent  | Good         | Needs Work  |
| ----------------- | ---------- | ------------ | ----------- |
| Final Val Loss    | < 0.02     | 0.02-0.05    | > 0.05      |
| Final Val MAE     | < 0.10     | 0.10-0.20    | > 0.20      |
| Training Time     | 5-15 min   | 15-30 min    | > 30 min    |
| Autopilot Success | 90%+ track | 70-90% track | < 70% track |

## Files Generated

After training, your session folder contains:

- `model.keras` - Final trained model
- `best_model.keras` - Best model from validation
- `training_log.csv` - Detailed metrics per epoch
- `training_plot.png` - Visual training progress
- `logs/` - TensorBoard logs (view with `tensorboard --logdir logs`)

## Next Steps

1. **Collect 500-1000 frames** of quality driving data
2. **Train with augmentation** and 50-100 epochs
3. **Check training plots** for good convergence
4. **Test at 0.5 throttle** first
5. **Gradually increase** to 0.7-0.8 if stable
6. **Iterate**: More data → Better model

Good luck! 🚗💨
