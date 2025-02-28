from ctgan import CTGAN
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np

class CustomCTGAN(CTGAN):
    def __init__(self, 
                 embedding_dim=128, 
                 generator_dim=(256, 256), 
                 discriminator_dim=(256, 256), 
                 generator_lr=0.0002, 
                 generator_decay=0.000001, 
                 discriminator_lr=0.0002, 
                 discriminator_decay=0.000001, 
                 batch_size=500, 
                 discriminator_steps=1, 
                 log_frequency=True, 
                 verbose=False, 
                 epochs=300, 
                 pac=10, 
                 cuda=True,
                 target_model=None, 
                 inv_criterion=None, 
                 alpha=1.0):
        
        super().__init__(embedding_dim, generator_dim, discriminator_dim, 
                         generator_lr, generator_decay, discriminator_lr, 
                         discriminator_decay, batch_size, discriminator_steps, 
                         log_frequency, verbose, epochs, pac, cuda)
        
        self.target_model = target_model
        self.inv_criterion = inv_criterion  # Invariance loss function
        self.alpha = alpha

    def train(self, data: pd.DataFrame, pseudo_labels: torch.Tensor, discrete_columns: list, epochs=None):
        """Train the CTGAN model using a PLG-MI-like attack setup.
        
        Args:
            data (pd.DataFrame): Data to train the model on.
            pseudo_labels (torch.Tensor): Pseudo-labels for each sample.
            discrete_columns (list): Column names for categorical variables.
            epochs (int): Number of training epochs.
        """
        device = torch.device("cuda" if self.cuda else "cpu")
        self.target_model.to(device)
        opt_gen = optim.Adam(self.generator.parameters(), lr=self.generator_lr, weight_decay=self.generator_decay)
        opt_dis = optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr, weight_decay=self.discriminator_decay)

        for epoch in range(epochs or self.epochs):
            for _ in range(self.discriminator_steps):
                # Sample real data
                idx = torch.randint(0, len(data), (self.batch_size,))
                real = torch.tensor(data.iloc[idx].values, dtype=torch.float32, device=device)
                real_labels = pseudo_labels[idx].to(device)

                # Generate fake data
                noise = torch.randn((self.batch_size, self.embedding_dim), device=device)
                fake_data = self.generator(noise)

                # Decode generated tabular data properly
                fake_df = self._decode(fake_data, discrete_columns)
                fake_df_tensor = torch.tensor(fake_df.values, dtype=torch.float32, device=device)

                # Assign pseudo-labels for the generated data
                fake_labels = torch.randint(0, real_labels.max() + 1, (self.batch_size,), device=device)

                # Compute discriminator outputs
                dis_real = self.discriminator(real)
                dis_fake = self.discriminator(fake_df_tensor)  # Use processed fake data

                # Discriminator loss
                loss_dis = F.binary_cross_entropy_with_logits(dis_real, torch.ones_like(dis_real)) + \
                           F.binary_cross_entropy_with_logits(dis_fake, torch.zeros_like(dis_fake))

                self.discriminator.zero_grad()
                loss_dis.backward()
                opt_dis.step()

            # Generator update
            fake_data = self.generator(noise)
            fake_df = self._decode(fake_data, discrete_columns)
            fake_df_tensor = torch.tensor(fake_df.values, dtype=torch.float32, device=device)

            dis_fake = self.discriminator(fake_df_tensor)

            # Compute invariance loss (ensuring generated samples match the target model’s feature space)
            inv_loss = self.inv_criterion(self.target_model(fake_df_tensor), fake_labels)

            # Generator loss
            loss_gen = F.binary_cross_entropy_with_logits(dis_fake, torch.ones_like(dis_fake))
            loss_total = loss_gen + self.alpha * inv_loss

            self.generator.zero_grad()
            loss_total.backward()
            opt_gen.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Gen Loss: {loss_gen.item():.4f}, Dis Loss: {loss_dis.item():.4f}, Inv Loss: {inv_loss.item():.4f}")

        print("Training complete!")

    def _decode(self, fake_data, discrete_columns, category_mappings):
        """Convert the generated numerical data back into categorical format using proper category mappings.

        Args:
            fake_data (torch.Tensor): Generated tabular data.
            discrete_columns (list): Names of categorical columns.
            category_mappings (dict): Mapping of categorical column names to their original categories.

        Returns:
            pd.DataFrame: Properly formatted tabular data.
        """
        fake_df = pd.DataFrame(fake_data.cpu().detach().numpy())

        for col in discrete_columns:
            col_idx = fake_df.columns.get_loc(col)  # Get column index
            num_categories = len(category_mappings[col])

            # Apply softmax over the categorical dimension
            logits = fake_df.iloc[:, col_idx : col_idx + num_categories].values
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # Softmax
            category_indices = np.argmax(probs, axis=1)  # Get the highest probability category

            # Map back to original categories
            fake_df[col] = [category_mappings[col][i] for i in category_indices]

            # Drop additional softmax-generated columns
            fake_df.drop(columns=fake_df.columns[col_idx : col_idx + num_categories], inplace=True)

        return fake_df

