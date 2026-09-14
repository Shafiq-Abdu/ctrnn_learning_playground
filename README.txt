CTRNN TEACHING PLATFORM v7
===========================

This package contains two browser modules:

1) dynamics/
   Explore a continuous-time recurrent neural network with fixed weights.
   Includes inputs, initial states, neural activity, PCA trajectory,
   fixed-point analysis and stability.

2) training/
   Train a small CTRNN on a delayed-response task using manual
   backpropagation through time (BPTT) written in JavaScript.

No Python, PyTorch, TensorFlow or backend server is required.

HOW TO RUN
----------
Recommended:
- Open the folder in VS Code.
- Use the Live Server extension on index.html.

You can also try opening index.html directly in a browser.

TRAINING MODEL
--------------
Architecture:
  1 input -> 4 recurrent tanh hidden units -> 1 linear output

Dynamics:
  h_(t+1) = h_t + (dt/tau)[-h_t + tanh(Wrec h_t + Win x_t + b)]

Readout:
  y_t = Wout h_(t+1) + bout

Task:
  input pulse: 1 to 2 s
  target pulse: 3 to 5 s
  total trial: 6 s
  dt = 0.1 s
  tau = 1.0 s

Loss:
  mean squared error over all time steps

Training:
  explicit/manual BPTT + global gradient clipping + SGD

IMPORTANT
---------
The training module is intentionally constrained for learning.
Later versions can expose hidden-unit count, task timing, activation,
optimizer, multiple training trials, noise and task families.


V7.1 COMPACT UPDATE
-------------------
The training module layout is compressed to reduce vertical scrolling. Core behavior and training code are unchanged.


V8 USER-CONTROLLED ARCHITECTURE
-------------------------------
Training module now allows:
- 1 to 3 input units
- 2, 3, 4, 5, 6, or 8 hidden recurrent units
- 1 to 3 output units

The delayed-response task is kept fixed:
- x1 receives the input pulse
- y1 carries the delayed target
- extra input channels are zero
- extra output targets are zero

Press "Apply architecture" after changing unit counts.
This reinitializes the weight matrices and resets training.


V8.1 INTERFACE + WEIGHT MATRICES
--------------------------------
- Polished training UI.
- Added live learned-parameter tables for Win, Wrec, Wout, hidden biases, and output biases.
- Matrix cells use sign/magnitude heat shading and update during training.


FINAL V1 NAMING + POLISH
------------------------
Project name: CTRNN Learning Playground

This release keeps the full functionality of the previous version:
- dynamics exploration
- manual JavaScript BPTT training
- SGD with gradient clipping
- user-controlled input/hidden/output unit counts
- live loss, hidden activity, PCA trajectory
- learned network visualization
- Win, Wrec, Wout and bias matrices
- compact browser-only interface

Terminology was cleaned up to avoid overusing "teaching"; the interface now uses "learning" or neutral wording instead.
