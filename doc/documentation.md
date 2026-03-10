# Doc

1. **Consructor**

   ```Python
   from ffnn.nn import MLP, Layer

   model = MLP([
      Layer(32, activation='relu', weight_init='he', l2=0.01),
      Layer(16, activation='relu', weight_init='he', l1=0.005),
      Layer(1, activation='sigmoid', weight_init='xavier')
   ], input_size=10, seed=42) # input_size=10 cascades down automatically!
   ```

2. **Dynamic Layers**

   ```Python
   model = MLP(input_size=10, seed=42)

   # Dynamically add layers
   for neurons in [32, 16, 8]:
      model.add(Layer(neurons, activation='swish', weight_init='he'))

   # Cap it off with the output layer
   model.add(Layer(3, activation='softmax', weight_init='xavier'))
   ```

3. **Compilation**

   a. String:

   ```Python
   # Uses default SGD parameters
   model.compile(optimizer='sgd', loss='mse')
   ```

   b. Hyperparameter:

   ```Python
   # Give specific optimizer settings
   model.compile(
      optimizer={'method': 'sgd', 'learning_rate': 0.05},
      loss='cce'
   )`
   ```

4. **Pipeline Example**

   ```Python
   from ffnn.nn import MLP, Layer
   import numpy as np

   # 1. Architecture
   model = MLP([
      Layer(64, activation='gelu', weight_init='he'),
      Layer(3, activation='softmax', weight_init='xavier')
   ], input_size=20)

   # 2. Compile
   model.compile(optimizer='sgd', loss='cce')

   # 3. Train
   history = model.fit(
      X_train, y_train,
      batch_size=32,
      epochs=100,
      learning_rate=0.01,
      validation_data=(X_val, y_val),
      verbose=1
   )

   # 4. Save for later deployment
   model.save("production_model.pkl")
   ```
