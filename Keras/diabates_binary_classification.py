#import libraries
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

#load the dataset 
dataset = loadtxt('pima-indians-diabetes.csv',delimiter=',')

#split into X and Y
X = dataset[:,:8]
Y = dataset[:,8]

# Define the model 
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model 
model.fit(X,Y, epochs=200, batch_size=10)

# Evaluate Keras Model
loss, accuracy = model.evaluate(X,Y)
print('Train Accuracy: %.2f'%(accuracy*100))


# Make predictions
# predictions = model.predict(X)
# rounded = [round(x[0]) for x in predictions]

# OR
predictions = model.predict_classes(X)
