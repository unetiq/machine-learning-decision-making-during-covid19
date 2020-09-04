import keras
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.models import Sequential


class RNNAutoencoder:
    """
    Wrapper for the recurrent neural autoencoder.

    :param optimizer: str; optimizer used for model training
    :param loss: str; loss type used for model training
    :param n_features: int; number of input features
    :param timesteps: int; number of data timestamps
    :param lstm_size: int; LSTM number of hidden units
    :param dense_size: int; Dense layer size
    """

    def __init__(self, optimizer, loss, n_features, timesteps, lstm_size, dense_size):
        self.optimizer = optimizer
        self.loss = loss
        self.n_features = n_features
        self.timesteps = timesteps
        self.lstm_size = lstm_size
        self.dense_size = dense_size

        self.__build_model__()

    def __build_model__(self):
        self.model = Sequential()

        # Encoder
        self.model.add(LSTM(self.lstm_size, activation="relu", input_shape=(self.timesteps, self.n_features)))
        self.model.add(Dense(self.dense_size, activation="relu"))
        self.model.add(RepeatVector(self.timesteps))

        # Decoder
        self.model.add(LSTM(self.lstm_size, activation="relu", return_sequences=True))
        self.model.add(TimeDistributed(Dense(self.n_features)))

        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model.summary()

    @property
    def fit(self, x, epochs, batch_size):
        """
        Fits the model on the given data using the given batch_size for the given epochs.

        :param x: np.array; training data
        :param epochs: int; number of epochs
        :param batch_size: int; batch size
        """
        self.model.fit(x, x, epochs=epochs, batch_size=batch_size)

    @property
    def encode(self, data):
        """
        Calculates the encoding for the given data by feeding it through the Encoder part.

        :param data: np.array; data to encode
        :return: encoded data
        """
        model_encode = keras.Model(self.model.input, self.model.layers[1].output)
        return model_encode(data)
