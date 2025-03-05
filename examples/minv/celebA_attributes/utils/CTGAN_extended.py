from ctgan import CTGAN
from ctgan.synthesizers.ctgan import Generator, Discriminator
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from torch import cuda, device

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
                 cuda=True):
        
        super().__init__(embedding_dim, generator_dim, discriminator_dim, 
                         generator_lr, generator_decay, discriminator_lr, 
                         discriminator_decay, batch_size, discriminator_steps, 
                         log_frequency, verbose, epochs, pac, cuda)

    def fit(self, train_data, target_model, num_classes, inv_criterion, gen_criterion, dis_criterion, alpha = 0.1, discrete_columns=()):
        """
        Fit the CTGAN model to the training data using pseudo-labeled guidance as in the PLG-MI attack.

        Args:
            train_data (pandas.DataFrame):
                Training data.
            target_model (torch.nn.Module):
                Target model.
            num_classes (int):
                Number of classes.
            inv_criterion (callable):
                Inversion criterion.
            gen_criterion (callable):
                Generator criterion.
            dis_criterion (callable):
                Discriminator criterion.
            alpha (float):
                Alpha value for the inversion loss.
            discrete_columns (list of str):
                List of column names that are discrete.
        """

        epochs = self._epochs
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = DataSampler(
            train_data, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), self._generator_dim, data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(), self._discriminator_dim, pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss', 'Inversion Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f}) | Inv. ({inv:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0, inv=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    """
                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt
                        )
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]
                    """
                    c1 = torch.randint(0, num_classes, (self._batch_size,), device=self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    real = self._data_sampler.sample_data(train_data, self._batch_size)
                    

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    #if c1 is not None:
                    #    fake_cat = torch.cat([fakeact, c1], dim=1)
                    #else:
                    real_cat = real
                    fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    # TODO: Maybe change this loss
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                """
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)
                """
                c1 = torch.randint(0, num_classes, (self._batch_size,), device=self._device)
                fakez = torch.cat([fakez, c1], dim=1)


                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                #if c1 is not None:
                #    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                #else:
                y_fake = discriminator(fakeact)

                # Pseudo label batch is last column of fakeact
                pseudo_label_batch = fakeact[:, -1].long()

                # Fake feature vector is all but last column of fakeact
                fakefeat = fakeact[:, :-1]

                if condvec is None:
                    inv_loss = 0
                else:
                    inv_loss  = inv_criterion(target_model(fakefeat), pseudo_label_batch)

                loss_g = gen_criterion(y_fake)
                loss_all = loss_g + inv_loss*alpha

                optimizerG.zero_grad(set_to_none=False)
                loss_all.backward()
                optimizerG.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()
            inversion_loss = inv_loss.detach().cpu().item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
                'Inversion Loss': [inversion_loss]
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )


