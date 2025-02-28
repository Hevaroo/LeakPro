from ctgan import CTGAN

class CustomCTGAN(CTGAN):
    def __init__(self, embedding_dim=128, generator_dim=..., discriminator_dim=..., generator_lr=0.0002, generator_decay=0.000001, discriminator_lr=0.0002, discriminator_decay=0.000001, batch_size=500, discriminator_steps=1, log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True):
        super().__init__(embedding_dim, generator_dim, discriminator_dim, generator_lr, generator_decay, discriminator_lr, discriminator_decay, batch_size, discriminator_steps, log_frequency, verbose, epochs, pac, cuda)
        
    def train(self, data, discrete_columns=tuple(), epochs=None):
        """Train the CTGAN model.
        
        Args:
            data (pd.DataFrame): Data to train the model on.
            discrete_columns (list of str): List of column names of discrete variables.
            epochs (int): Number of training epochs.
        """
        super().fit(data, discrete_columns, epochs)
        