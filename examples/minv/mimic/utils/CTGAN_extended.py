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
from sdv.evaluation.single_table import evaluate_quality, get_column_plot
from sdv.metadata import Metadata

class CustomCTGAN(CTGAN):
    def __init__(self, 
                 embedding_dim=256, 
                 generator_dim=(256, 256), 
                 discriminator_dim=(256, 256), 
                 generator_lr = 2e-4,
                 generator_decay=1e-6,
                 discriminator_lr=2e-5,
                 discriminator_decay=1e-7,
                 num_classes=5088,
                 batch_size=500, 
                 discriminator_steps=5,
                 log_frequency=True, 
                 verbose=False, 
                 epochs=300, 
                 pac=10, 
                 cuda=True):
        
        self.dim_z = embedding_dim

        super().__init__(embedding_dim, generator_dim, discriminator_dim, 
                         generator_lr, generator_decay, discriminator_lr, 
                         discriminator_decay, batch_size, discriminator_steps, 
                         log_frequency, verbose, epochs, pac, cuda)
        
        self._transformer = DataTransformer()

        self.metadata = None
        
    def to(self, device):
        self._device = device
        self._generator.to(device)

    
    def eval(self):
        self._generator.eval()

        
    def __call__(self, z=None, y=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if self._transformer is None:
            raise ValueError("The transformer has not been initialized. Please call the `fit` method first.")

        condition_column = "pseudo_label"
    
        if y is not None:
            # Batch size is length of y
            bs = y.shape[0]
            condition_values = y.detach().cpu().numpy()

            # TODO: Check this logic
            condition_values = condition_values[:1]
            
        else:
            bs = z.shape[0]
            condition_values = [None] * z.shape[0]


        samples = pd.DataFrame()
        
        for condition_value in condition_values:        
            if condition_column is not None and condition_value is not None:
                try:
                    condition_info = self._transformer.convert_column_name_value_to_id(
                        condition_column, condition_value
                    )
                    global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                        condition_info, bs
                    )
                except ValueError:
                    # If the transformer has not seen the condition value in training, it will raise a ValueError
                    # We still want to be able to sample, so we set the global_condition_vec to None
                    global_condition_vec = None
            else:
                global_condition_vec = None

            data = []
            if z is None:
                mean = torch.zeros(bs, self._embedding_dim)
                std = mean + 1
                fakez = torch.normal(mean=mean, std=std).to(self._device)
            else:
                fakez = z
            
            
            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(bs)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

            data = np.concatenate(data, axis=0)

            sample = self._transformer.inverse_transform(data)
            # add row to samples
            samples = pd.concat([samples, sample])
        return samples
    
    def fit2(self, train_data, target_model, num_classes, inv_criterion, gen_criterion, dis_criterion, n_iter, n_dis, alpha = 0.1, discrete_columns=(), use_inv_loss=True):
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
        epochs = n_iter
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)

        # Convert "pseudo_label" column to categorical
        train_data["pseudo_label"] = train_data["pseudo_label"].astype("category")

        original_train_data = train_data.copy()

        #self.metadata = Metadata.detect_from_dataframe(data=original_train_data,table_name='mimic')

        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)
        
        self.num_classes = num_classes

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
            #weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            #weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss', 'Inversion Loss'])
        
        
        # num_unused_discrete_categories = 0
        # for col in discrete_columns:
        #     if col != "pseudo_label":
        
        #         # Get categories in col
        #         categories = original_train_data[col].unique()
        #         num_unused_discrete_categories += len(categories)
        #         for value in categories:
        #             info = self._transformer.convert_column_name_value_to_id(col, value)
        #             print( f"Column: {col}, Value: {value}, Info: {info}")

        #             c1 = self._data_sampler.generate_cond_from_condition_column_info(info, 1)
        #             # find index of the fist 1 in the condition vector
        #             index = np.where(c1[0] == 1)[0][0]
        #             print(f"Index: {index}")

        def sample_pseudo_c1(condition_values):
            """
            Sample conditional vector c1 with randomized condition values for each sample in the batch.
            """
            condition_column = "pseudo_label"
            
            # Generate random condition values for each sample in the batch
           
            
            try:
                warnings.filterwarnings('ignore') 

                # Generate condition vectors for all condition values in the batch
                condition_info_list = [
                    self._transformer.convert_column_name_value_to_id(condition_column, value)
                    for value in condition_values
                ]

                global_condition_vec = np.vstack([
                    self._data_sampler.generate_cond_from_condition_column_info(info, 1)
                    for info in condition_info_list
                ])
            except ValueError:
                # If the transformer has not seen some condition values in training, it will raise a ValueError
                # We still want to be able to sample, so we set the global_condition_vec to None
                global_condition_vec = None

            return global_condition_vec
        
        def sample_pseudo_c1_fast(condition_values):
            global_cond_vec = []
            for value in condition_values:
                cond_vec = np.zeros((1, self._data_sampler.dim_cond_vec()))
                cond_vec[0, value] = 1
                global_cond_vec.append(cond_vec)
            global_cond_vec = np.vstack(global_cond_vec).astype('float32')
            return global_cond_vec


        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.4f}) | Dis. ({dis:.4f}) | Inv. ({inv:.4f}) | CE. ({c_loss:.4f}) | Acc. ({acc:.4f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0, inv=0, c_loss=0, acc=0))

        # Set constant mask to always condition on pseudo-label
        pl_condition_info = self._transformer.convert_column_name_value_to_id(
                        'pseudo_label', 0
                    )
        m1 = np.zeros(len(discrete_columns))
        m1[pl_condition_info['discrete_column_id']] = 1
        m1 = [m1]*self._batch_size
        m1 = torch.from_numpy(np.array(m1)).to(self._device)

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):
                for n in range(n_dis):
                    
                    # TODO: Only condition on pseudo-labels
                    
                    fakez = torch.normal(mean=mean, std=std)

                    # Sample condition values for the batch from original (not transformed) train_data
                    condition_values = original_train_data["pseudo_label"].sample(self._batch_size).values
                    # Sample conditional vector c1 from the condition values (pseudo-labels)
                    c1 = sample_pseudo_c1_fast(condition_values)


                    # for row in c1:
                    #     # Find index of non zero element in row
                    #     index = np.where(row == 1)[0][0]
                    #     print(f"Index: {index}")

                        
                    if c1 is None:
                        c1 = self._data_sampler.sample_original_condvec(self._batch_size)   
                        print("SAMPLING FROM ORIGINAL COND VEC")         

                    # We want to sample the same condition values for the real data as we did for the fake data
                    # They should be mismatched with real sampled data for discriminator training

                    # Permute condition values and c1 to get c2, we need
                    perm = np.arange(self._batch_size)
                    np.random.shuffle(perm)

                    col_idx = discrete_columns.index("pseudo_label")

                    # Create column indices list
                    col = [col_idx] * self._batch_size

                    # Sample real data
                    real = self._data_sampler.sample_data(train_data, self._batch_size, col=col, opt=condition_values[perm])

                    c2 = c1[perm]
                    c2 = torch.from_numpy(c2).to(self._device)
                    
                    c1 = torch.from_numpy(c1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)
                    #real_copy = real.copy()

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact
                    
                    
                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    # TODO: Maybe change this loss
                    #loss_d = F.relu(1. - y_real).mean() + F.relu(1. + y_fake).mean()
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)
                
                # Sample condition values for the batch from original train_data
                condition_values = original_train_data["pseudo_label"].sample(self._batch_size).values
                # Sample conditional vector c1 from the condition values (pseudo-labels)
                c1 = sample_pseudo_c1_fast(condition_values)

                if c1 is None:                   
                    c1 = self._data_sampler.sample_original_condvec(self._batch_size)
                    print("SAMPLING FROM ORIGINAL COND VEC, GEN STEP")
                
                c1 = torch.from_numpy(c1).to(self._device)
                
                fakez = torch.cat([fakez, c1], dim=1)
                   
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                # Fake feature vector
                fakefeat = fakeact
                
                fakefeat = fakefeat.detach().cpu().numpy()
                     
                sample = self._transformer.inverse_transform(fakefeat)

                #sample_copy = sample.copy()
                pseudo_label_batch = sample['pseudo_label'].values
                pseudo_label_batch = torch.tensor(pseudo_label_batch, device=self._device)
                #remove 'pseudo_label' column
                sample = sample.drop(columns=['pseudo_label'])                
            
                if condvec is None or not use_inv_loss:
                    inv_loss = 0
                else:
                    inv_loss  = inv_criterion(target_model(sample), pseudo_label_batch)

                # Define m1, the mask for the condition vector
                # Since we are always condit
                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                beta = 0.5
                loss_g = gen_criterion(y_fake)
                loss_all = loss_g + inv_loss*alpha + beta*cross_entropy

                optimizerG.zero_grad(set_to_none=False)
                loss_all.backward()
                optimizerG.step()

            #real_inverse = self._transformer.inverse_transform(real_copy)

            # 2. measure the statistical similarity
            #quality_report = evaluate_quality(real_inverse, sample_copy, self.metadata)
            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()
            if use_inv_loss:
                inversion_loss = inv_loss.detach().cpu().item()
            else:
                inversion_loss = 0

            with torch.no_grad():
                # Compute accuracy
                count = sample.shape[0]
                T_logits = target_model(sample)
                T_preds = T_logits.max(1)[1]
                acc = (T_preds == pseudo_label_batch).sum() / count
                acc = acc.item()

            cross_entropy = cross_entropy.detach().cpu().item()
            
            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
                'Inversion Loss': [inversion_loss],
                'Conditioning Loss (CE)': [cross_entropy],
                'Accuracy' : [acc]
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss, inv=inversion_loss,c_loss=cross_entropy, acc=acc)
                )

    def fit(self, train_data, target_model, num_classes, inv_criterion, gen_criterion, dis_criterion, n_iter, n_dis, alpha = 0.1, discrete_columns=(), use_inv_loss=True):
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
        epochs = n_iter
        self._validate_discrete_columns(train_data, discrete_columns)
        self._validate_null_data(train_data, discrete_columns)

        # Convert "pseudo_label" column to categorical
        train_data["pseudo_label"] = train_data["pseudo_label"].astype("category")
        #self.metadata = Metadata.detect_from_dataframe(data=original_train_data,table_name='mimic')

        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)
        
        self.num_classes = num_classes

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

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss', 'Inversion Loss', 'C_loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Dis. ({dis:.2f}) | Inv. ({inv:.2f}) | CE. ({c_loss:.2f}) | Acc. ({acc:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0, inv=0, c_loss=0, acc=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):
                for n in range(n_dis):
                    
                    # TODO: Only condition on pseudo-labels
                    
                    fakez = torch.normal(mean=mean, std=std)

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

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact
                    
                    
                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    # TODO: Maybe change this loss
                    #loss_d = F.relu(1. - y_real).mean() + F.relu(1. + y_fake).mean()
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)
           
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                   
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                # Pseudo label batch is last column of fakeact
                #pseudo_label_batch = torch.from_numpy(condition_values).to(self._device)
                
                # We want the pseudo_label_batch to be the same as the condition values

                # Fake feature vector
                fakefeat = fakeact
                
                fakefeat = fakefeat.detach().cpu().numpy()
                     
                sample = self._transformer.inverse_transform(fakefeat)
                #sample_copy = sample.copy()
                pseudo_label_batch = sample['pseudo_label'].values
                pseudo_label_batch = torch.tensor(pseudo_label_batch, device=self._device)
                #remove 'pseudo_label' column
                sample = sample.drop(columns=['pseudo_label'])                
                
                # if condvec is None or not use_inv_loss
                if condvec is None or not use_inv_loss:
                    inv_loss = 0
                else:
                    inv_loss  = inv_criterion(target_model(sample), pseudo_label_batch)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                beta = 0.3
                # TODO: check c1 perhaps p_label should be replaced with sample label, CE
                loss_g = gen_criterion(y_fake)
                loss_all = loss_g + inv_loss*alpha + beta*cross_entropy 

                optimizerG.zero_grad(set_to_none=False)
                loss_all.backward()
                optimizerG.step()

            #real_inverse = self._transformer.inverse_transform(real_copy)

            # 2. measure the statistical similarity
            #quality_report = evaluate_quality(real_inverse, sample_copy, self.metadata)
            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()
            if use_inv_loss:
                inversion_loss = inv_loss.detach().cpu().item()
            else:
                inversion_loss = 0

            with torch.no_grad():
                # Compute accuracy
                count = sample.shape[0]
                T_logits = target_model(sample)
                T_preds = T_logits.max(1)[1]
                acc = (T_preds == pseudo_label_batch).sum() / count
                acc = acc.item()

            cross_entropy = cross_entropy.detach().cpu().item()
            
            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss],
                'Inversion Loss': [inversion_loss],
                'Conditioning Loss (CE)': [cross_entropy],
                'Accuracy' : [acc]
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss, inv=inversion_loss,c_loss=cross_entropy, acc=acc)
                )
            
        