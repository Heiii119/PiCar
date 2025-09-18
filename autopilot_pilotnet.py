"""
autopilot_pilotnet.py

Lightweight PilotNet-style autopilot module that reuses camera and motor helpers
from drive_train_autopilot_picam2.py in this workspace.

Features:
- build PilotNet model
- tf.data augmentation (flip, brightness/contrast)
- training harness saving .h5 model
- optional TFLite export
- runtime loop to run autopilot with keyboard override

Usage examples are in AUTOPILOT_README.md
"""
import os
import time
import argparse
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
except Exception as e:
    raise ImportError("TensorFlow is required for autopilot_pilotnet.py: " + str(e))

# Reuse plumbing from existing script
from drive_train_autopilot_picam2 import (
    PiCam2Manager,
    MotorServoController,
    load_image_for_model,
    load_dataset,
    IMAGE_H,
    IMAGE_W,
    IMAGE_DEPTH,
    DRIVE_LOOP_HZ,
    PWM_STEERING_THROTTLE,
    RawKeyboard,
    KeyboardDriver,
)


def make_tf_dataset(X, y, batch_size=32, augment=True, shuffle=True):
    """Create a tf.data.Dataset from numpy arrays with optional augmentation.
    Flipping horizontally will invert the steering label (y[...,0]).
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))

    def _prep(img, lbl):
        img = tf.cast(img, tf.float32) / 255.0
        steer = lbl[0]
        thr = lbl[1]

        if augment:
            do_flip = tf.random.uniform([], 0, 1) > 0.5
            img = tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img)
            steer = tf.cond(do_flip, lambda: -steer, lambda: steer)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)

        out_lbl = tf.stack([steer, thr])
        return img, out_lbl

    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_pilotnet(input_shape=(IMAGE_H, IMAGE_W, IMAGE_DEPTH)):
    """PilotNet / NVIDIA-inspired small network returning [steer, throttle]."""
    inp = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255.0)(inp)
    x = layers.Conv2D(24, (5, 5), strides=2, activation='relu')(x)
    x = layers.Conv2D(36, (5, 5), strides=2, activation='relu')(x)
    x = layers.Conv2D(48, (5, 5), strides=2, activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(50, activation='relu')(x)
    x = layers.Dense(10, activation='relu')(x)
    out = layers.Dense(2, activation='tanh')(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(1e-4), loss='mse', metrics=['mae'])
    return model


def train(session_root, batch_size=32, epochs=30, export_tflite=False):
    if session_root is None or not os.path.exists(session_root):
        print("Session root not found.")
        return None

    X, y = load_dataset(session_root)
    if len(X) < 50:
        print("Not enough samples to train (need ~50+).")
        return None

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    n = len(X)
    n_train = int(0.8 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    model = build_pilotnet((IMAGE_H, IMAGE_W, IMAGE_DEPTH))

    train_ds = make_tf_dataset(X_train, y_train, batch_size=batch_size, augment=True, shuffle=True)
    val_ds = make_tf_dataset(X_val, y_val, batch_size=batch_size, augment=False, shuffle=False)

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model_path = os.path.join(session_root, 'model_pilotnet.h5')
    model.save(model_path)
    print(f"Saved Keras model to {model_path}")

    # Save training history to JSON
    try:
        history_dict = history.history if hasattr(history, 'history') else dict()
        import json
        hist_path = os.path.join(session_root, 'history.json')
        with open(hist_path, 'w') as hf:
            json.dump(history_dict, hf, indent=2)
        print(f"Saved training history to {hist_path}")
    except Exception as e:
        print("Failed to save history JSON:", e)

    # Plot history if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        if 'loss' in history_dict:
            plt.plot(history_dict.get('loss', []), label='loss')
        if 'val_loss' in history_dict:
            plt.plot(history_dict.get('val_loss', []), label='val_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.twinx()
        # plot mae on secondary axis if present
        if 'mae' in history_dict:
            plt.plot(history_dict.get('mae', []), '--', color='orange', label='mae')
        if 'val_mae' in history_dict:
            plt.plot(history_dict.get('val_mae', []), '--', color='red', label='val_mae')
        plt.title('Training history')
        plt.tight_layout()
        hist_img = os.path.join(session_root, 'history.png')
        plt.savefig(hist_img)
        plt.close()
        print(f"Saved training plot to {hist_img}")
    except Exception as e:
        print("Matplotlib not available or plotting failed:", e)

    if export_tflite:
        tflite_path = os.path.join(session_root, 'model_pilotnet.tflite')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        try:
            tflite_model = converter.convert()
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            print(f"Exported TFLite model to {tflite_path}")
        except Exception as e:
            print("TFLite export failed:", e)
    return model_path


def _predict_keras(model, img_rgb):
    inp = np.expand_dims(load_image_for_model(img_rgb).astype(np.float32) / 255.0, axis=0)
    pred = model.predict(inp, verbose=0)[0]
    return float(np.clip(pred[0], -1, 1)), float(np.clip(pred[1], -1, 1))


def _predict_tflite(interpreter, input_details, output_details, img_rgb):
    inp = load_image_for_model(img_rgb).astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])[0]
    return float(np.clip(out[0], -1, 1)), float(np.clip(out[1], -1, 1))


def run_autopilot(model_path, use_tflite=False, with_preview=True):
    if model_path is None or not os.path.exists(model_path):
        print("Model not found; cannot run autopilot.")
        return

    if use_tflite:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        predict_fn = lambda img: _predict_tflite(interpreter, input_details, output_details, img)
        print("Running with TFLite interpreter.")
    else:
        model = tf.keras.models.load_model(model_path)
        predict_fn = lambda img: _predict_keras(model, img)
        print("Running with Keras model.")

    cam = PiCam2Manager(IMAGE_W, IMAGE_H, DRIVE_LOOP_HZ, with_preview=with_preview)
    ctrl = MotorServoController(PWM_STEERING_THROTTLE)
    period = 1.0 / DRIVE_LOOP_HZ
    last_loop = time.time()

    manual_override = False
    driver = KeyboardDriver()

    # Arm
    ctrl.stop()
    time.sleep(1.0)

    try:
        with RawKeyboard() as global_kb:
            global kb
            kb = global_kb
            print("Autopilot running. h=manual, a=auto, q=quit.")
            while True:
                frame_rgb = cam.capture_rgb()
                ch = kb.get_key(timeout=0.0)
                if ch == 'q':
                    break
                if ch == 'h':
                    manual_override = True
                elif ch == 'a':
                    manual_override = False
                else:
                    if manual_override:
                        driver.handle_char(ch)

                if not manual_override:
                    steer, thr = predict_fn(frame_rgb)
                else:
                    steer = driver.steering
                    thr = driver.throttle

                ctrl.set_steering(steer)
                ctrl.set_throttle(thr)

                now = time.time()
                dt = now - last_loop
                if dt < period:
                    time.sleep(period - dt)
                last_loop = time.time()
    finally:
        try:
            cam.stop()
        except Exception:
            pass
        ctrl.stop()
        ctrl.close()


def _cli():
    parser = argparse.ArgumentParser(description='Autopilot PilotNet helper')
    sub = parser.add_subparsers(dest='cmd')

    p_train = sub.add_parser('train')
    p_train.add_argument('session_root')
    p_train.add_argument('--epochs', type=int, default=30)
    p_train.add_argument('--batch', type=int, default=32)
    p_train.add_argument('--tflite', action='store_true')

    p_run = sub.add_parser('run')
    p_run.add_argument('model_path')
    p_run.add_argument('--tflite', action='store_true')
    p_run.add_argument('--no-preview', action='store_true')

    args = parser.parse_args()
    if args.cmd == 'train':
        train(args.session_root, batch_size=args.batch, epochs=args.epochs, export_tflite=args.tflite)
    elif args.cmd == 'run':
        run_autopilot(args.model_path, use_tflite=args.tflite, with_preview=not args.no_preview)
    else:
        parser.print_help()


if __name__ == '__main__':
    _cli()
