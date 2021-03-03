model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'),
                        input_shape=(None, n_length, num_features)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(Dropout(0.4)))
model.add(TimeDistributed(MaxPooling1D(1)))
model.add(TimeDistributed(Flatten()))
model.add(BatchNormalization())
model.add(LSTM(1024))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(number_labels, activation='softmax'))

optimizer = keras.optimizers.Adam(lr=0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                metrics=['categorical_accuracy'])

print(model.summary())
