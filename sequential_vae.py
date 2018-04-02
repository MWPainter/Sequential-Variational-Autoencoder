from abstract_network import *
from scipy import misc
from tf.contrib import layers


def step(x):
    # x: tf tensor
    # return 0 if x < 0, return 1 if x >= 0
    return (tf.sign(x) + 1.0) / 2.0

def threshold(x, eps):
    # x: tf tensor
    # eps: float threshold
    # return 1 if x >= eps, which is true iff x/eps - 1 >= 0, return 0 otherwise
    return step(x/eps - 1.0)


class SequentialVAE(Network):
    """
    Implementation of SequentialVAE, extending our abstract Network class. SequentialVAE essentially uses a VAE to 
    encode a Markov Chain to form a generative model. We consider a Makov Chain of random variables z_0, x_1, z_1, x_2, 
    z_2, ..., used in our generation process, where z_0 is sampled from a unit Gaussian and the subsequent steps are 
    encoded by the 

    Sequential VAE consists of networks encoding z'_i = g_theta(x_i), q_phi(z''_i | x), p_theta(x_i+1 | z_i), where 
    z_i is some function (usually concatenation) of z'_i and z''_i. We will refer to the network for g_theta as the 
    'encoder', the network for q_phi as the 'latent_predictor' or sometimes the 'recognition_network' (if there's a 
    dependency on x), and p_theta we refer to as the 'decoder' or 'generator'.

    So, letting z_i = f(z'_i, z''_i) we have q_phi(z_i | x_i, x) = q_phi(f(z'_i, z''_i) | x_i, x) = q_phi(z''_i | x).
    At generation time, we replace q_phi(z''_( | x) by the prior q_phi(z''_i) = N(0,1), enforced by a regularization 
    loss

    The chain z_0, x_1, z_1, x_2, z_2, ... is sampled by z_i ~ q_phi(z_i | x_i) and x_i ~ p_theta(x_i | z_i-1).
    At train time we use q_phi(z''_i | x) for training and replace it by q_phi(z''_i) = N(0,1) at generation time.
    All probability distributions are (diagonal) Gaussian, and completely encoded by the mean's and stddev's in each 
    dimension. (Maybe it would be interesting to try non gaussian)

    Also note, in the current implementation, that q_phi(z'_i | x_i) is degenrate, and we actually use a deterministic 
    function, i.e. some function g_theta of the form z'_i = g)theta(x_i).

    Currently the code uses the "inference_ladder" and "generative_ladder" which implement the VLAE architecture. 
    However, we just use it as a typical (variational) auto-encoder

    diagram:
          g         p
    x_i ----> z_i ----> z_i+1
               ^
               |  q
               x

    Moreover, we have a number of variants coded by this model, as described here:
    1. inhomogeneous MC
    In this case phi and theta are dependent on the timestep (in the MC), and so are encoded by different networks. In 
    this setting we can only train MCs of a fixed length.

    2. homogeneous MC
    Phi and theta are not dependent on the timestep.

    3. Latent InfoMax 
    We replace q_phi(z''_i | x) by q_phi(z''_i | x_i-1), which changes the graphical model. Also, we introduce a new 
    objective to train q_phi(z''_i | x_i), outlined in the paper. We still use q_phi(z_0 | x) for the first step when 
    training. Note that the only difference between the training and denerative modes is now 
    q_phi(z_0) vs q_phi(z_0 | x).
    """
    def __init__(self, dataset, batch_size, name, logger):
        """
        Initialization of the network, defines parameters and constructs the network in tenforflow
        SEE ALSO: __init__ from abstract_network.py, which defines our superclass 'Network'

        Description of parameters defined:
        (VLAE params)
        self.vlae_levels = the number of levels in each VLAE used in the seqvae. (Technically, "one level" here is two 
                    convolutional layers, each level reduces the image size by half (step sizes of 2 then 1)
                    (the last level has a fully connected layer too)
        self.vlae_latent_dims = a list of the number of dimensions in the latent space for each part of the ladder in 
                    the vlae
        self.image_sizes = the size of the "intermediate images". A list of 'self.vlae_levels + 1' integers.
        self.filter_sizes = the size of the filters. A list of length 'self.vlae_levels + 2', as there are 
                    'self.VLAE_levels + 1' intermediate representations (including input) and we have two for the 
                    output level (includes a fully connected layer)

        (SeqVAE params)
        self.share_theta_weights = indicates any networks which include weights as part of 'theta', should share 
                    parameters/weights. (I.e. if this is true, g_theta and p_theta should always be the same at every 
                    MC timestep). We could also think of this as making the 'theta' parts of the sequential vae be 
                    time homogeneous
        self.share_phi_weights = indicates any networks which include weights as part of 'phi', should share 
                    parameters/weights. (I.e. if this is true, q_phi should always be the same at every MC timestep). 
                    We could also think of this as making the 'phi' pars for the sequential vae be time homogeneous
        self.inference = the recognition network to use (currently just VLAE's recognition network is an option)
                    N.B. self.inference is a functional value
        self.generator = the generator network to use (currently just VLAE's generator is an option)
                    N.B. self.generator is a functional value
        self.mc_steps = the number of steps to use in the markov chain
        self.latent_dim = the dimension of the latent spaces (i.e. the number of dimensions z_i has for EACH i)
        self.intermediate_reconstruction = if each x_i should be a reconstruction of x. (If we should add a 
                    reconstruction loss for each x_i)
        self.early_stopping_threshold = threshold on the L2 distance of successive samples for which we stop the chain,
                    note that we only stop early if 'self.homogeneous_operation' is true
        self.homogeneous_operation = if we want to run the network in a homogenous mode. This makes the network a single 
                    step long, and constructed using 'self.construct_network_homogeneous', rather than 
                    'self.construct_network'. *** trainer.py treats the network differently in this case ***.
                    See trainer.py and 'self.construct_network_homogeneous' for more details
        self.predict_latent_code = true if we want to run the "Latent InfoMax" version of the MC. Can be used with either 
                    homgogeneous or inhomeogeneous operation. This predicts the latent code using x_{t-1} rather than x 
                    (in BOTH gnerative and training samples).
        self.combine_noise_method = 'concat'/'add'/'gated_add', and specifies how to add in (latent) noise into the 
                    embeddings of the autoencoder (see combined_noise for more detail)

        'self.combine_noise_method' should take one of the following values:
        "concat": Concatenate the noise onto the embeddings
        "add": directly add the noise and the embeddings
        "gated_add": multiply the noise by a (trainable) variable, and then add

        (Hyperparams)
        self.learning_rate = the learning rate to use during training
        self.reg_coeff_rate = decides the rate of which we increase the (coefficient of) regularization on latent 
                    variables from 0 to 1 over time. (It's an inverse exponential, 1-e^-t)

        (Training params)
        self.save_freq = how frequently to save the network during training
        self.tb_summary_freq = how frequently to write summaries to tensorboard
        self.add_debug_tb_variables = if we should add tensorboard variables which are only necessary for debugging

        :param dataset: the dataset that we will be training this network on 
                        (implicitly defining the p_data we wish to model)
        :param batch_size: the batch size to use during training
        :param name: the name we are allocating this instand of the seqvae network 
        :param logger: a logger object, for logging 
        :return: None
        """
        Network.__init__(self, dataset, logger)

        self.dataset = dataset
        self.batch_size = batch_size
        self.data_dims = dataset.data_dims
        self.name = name
        self.LOG = logger

        # VLAE parameters - assumes input image is square and at least a multiple of 16 (usually power of 2)
        self.vlae_levels = 4
        self.vlae_latent_dims = [10, 10, 10, 10]
        self.image_sizes = [self.data_dims[0], self.data_dims[0] // 2, 
                            self.data_dims[0] // 4, self.data_dims[0] // 8,
                            self.data_dims[0] // 16]
        self.filter_sizes = [self.data_dims[-1], 24, 48, 96, 192, 384]

        # SeqVAE parameters 
        self.share_theta_weights = False
        self.share_phi_weights = False
        self.inference = self.inference_ladder 
        self.generator = self.generator_ladder 
        self.mc_steps = 8
        self.latent_dim = np.sum(self.vlae_latent_dims)
        self.intermediate_reconstruction = True
        self.early_stopping_threshold = 0.000005
        self.homogeneous_operation = False
        self.predict_latent_code = False
        self.combine_noise_method = "concat"

        # Hyperparams and training params
        self.learning_rate = 0.0002
        self.reg_coeff_rate = 5000.0 # 1 epoch = 1000
        self.save_freq = 2000
        self.tb_summary_freq = 10
        self.add_debug_tb_variables = True

        # Config for different netnames, where customization is needed.
        # add overides for any of the above parameters here
        if self.name == "sequential_vae_celebA_inhomog":
            # nothing
            pass

        elif self.name == "sequential_vae_celebA_homog":
            self.share_theta_weights = True

        # elif self.name == "sequential_vae_celebA_homog_early_stopping":
        #     self.share_theta_weights = True
        #     self.mc_steps = 15

        elif self.name == "sequential_vae_celebA_inhomog_inf_max":
            self.predict_latent_code = True

        elif self.name == "sequential_vae_lsun":
            self.vlae_latent_dims = [20, 30, 30, 30]
            self.latent_dim = np.sum(self.vlae_latent_dims)

        elif self.name == "sequential_vae_lsun_single":
            self.vlae_latent_dims = [160, 240, 240, 240]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.mc_steps = 1

        elif self.name == "sequential_vae_lsun_final":
            self.vlae_latent_dims = [20, 30, 30, 30]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 32, 64, 128, 384, 512]
            self.intermediate_reconstruction = False

        # Latent pred not yet refactored back in
        # elif self.name == "sequential_vae_lsun_pred":
        #     self.filter_sizes = [self.data_dims[-1], 64, 128, 256, 512, 1024]
        #     self.vlae_latent_dims = [20, 20, 20, 20]
        #     self.latent_dim = np.sum(self.ladder_dims)
        #     self.use_latent_pred = True

        # python main.py --dataset=mnist --plot_reconstruction --netname=sequential_vae_mnist_inhomog
        elif self.name == "sequential_vae_mnist_inhomog":
            self.vlae_levels = 3
            self.vlae_latent_dims = [8, 8, 8]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]
            self.mc_steps = 5

        elif self.name == "sequential_vae_mnist_homog":
            self.vlae_levels = 3
            self.vlae_latent_dims = [8, 8, 8]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]
            self.mc_steps = 5
            self.share_theta_weights = True

        elif self.name == "sequential_vae_mnist_homog_early_stopping":
            self.vlae_levels = 3
            self.vlae_latent_dims = [8, 8, 8]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]
            self.mc_steps = 15
            self.share_theta_weights = True

        elif self.name == "sequential_vae_mnist_share_all":
            self.vlae_levels = 3
            self.vlae_latent_dims = [8, 8, 8]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]
            self.mc_steps = 5
            self.share_theta_weights = True
            self.share_phi_weights = True

        elif self.name == "sequential_vae_mnist_share_encoders":
            self.vlae_levels = 3
            self.vlae_latent_dims = [8, 8, 8]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]
            self.mc_steps = 5
            self.share_encoder_params = True

        elif self.name == "sequential_vae_mnist_share_inference":
            self.vlae_levels = 3
            self.vlae_latent_dims = [8, 8, 8]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]
            self.mc_steps = 5
            self.share_phi_weights = True

        elif self.name == "sequential_vae_mnist_inhomog_inf_max":
            self.vlae_levels = 3
            self.vlae_latent_dims = [8, 8, 8]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]
            self.mc_steps = 5
            self.predict_latent_code = True

        else:
            self.LOG.error("Unknown network name %s" % self.name)
            exit(-1)
    
        # Construct initialize and print network
        if not self.homogeneous_operation:
            self.construct_network()
        else:
            self.construct_network_homogeneous()
        self.init_network()
        self.print_network()
        self.log_tf_variables()

        # stuff for figures
        self.mc_fig, self.mc_ax = None, None





    def construct_network(self):
        """
        Construct the network, using 'self.inference' and 'self.generator'. This network will produce
        the sequence of samples and latent random variables.

        Tensorflow variables defined here:
        self.input_placeholder = placeholder for the input image (= the target image + optional noise)
        self.target_placeholder =  placeholder for the target image (= input image, without any noise)
        self.reg_coeff = placeholder (with default value) for the coefficient of regularization
        
        self.latents = placeholders for latent variables, use these when we want to use the network generatively
        self.training_samples = samples generated along the chain, use these one's when training the network this uses a 
                    latent variable that's sampled using the mean and stddev from inference network of the previous 
                    sample
        self.generator_samples = samples generated along the chaing, use these one's when running the generatively this 
                    uses the latent variable fed into the placeholder in self.latents

        N.B. Although when making the generator and training sample variables, we make two seperate 
        calls to 'self.generator', parameter sharing (through variable scopes) 
    
        self.loss = the total loss, the value that sums all of the lasses from each stage in the network and which we 
                    optimize over
        self.final_loss = the loss of the final sample (i.e. the output of the markov chain). i.e. how good our 
                    performance is

        self.merged_summaries = a tensorflow op, which combines all previous tensorboard summaries, so we can run all 
                    of them at the same time (in a tf session)
        self.train_op = the tensorflow training op
        """
        self.input_placeholder = tf.placeholder(shape=[None]+self.data_dims, dtype=tf.float32, name="input_placeholder")
        self.target_placeholder = tf.placeholder(shape=[None]+self.data_dims, dtype=tf.float32, name="target_placeholder")
        self.reg_coeff = tf.placeholder_with_default(1.0, shape=[], name="regularization_coefficient")

        self.latents = []
        self.training_samples = []
        self.generator_samples = []

        self.loss = 0.0
        self.final_loss = None

        training_sample = None
        generator_sample = None

        for step in range(self.mc_steps):
            # Create placeholder for latent variable on this step. i.e. z_i (used in "generative mode")
            latent_placeholder = tf.placeholder(shape = [None, self.latent_dim], 
                                                dtype = tf.float32, 
                                                name = ("latent_placeholder_%d" % step))
            self.latents.append(latent_placeholder)

            # On the first step, let x_0 be uniform random noise
            if step == 0:
                generator_sample = tf.random_uniform(shape=tf.stack([tf.shape(self.input_placeholder)[0]] + self.data_dims))
                self.generator_samples.append(generator_sample)

            # Make recognition, p_phi(z_t|x), and generative, p_theta(x_t|z_t,x_t-1), networks. Append samples from them
            latent_mean, latent_stddev, latent_train, latent_generative = self.create_recognition_network(step=step)
            training_sample, generator_sample = self.create_generator_network(training_sample, generator_sample, 
                                                                              latent_train, latent_generative, step)
            self.training_samples.append(training_sample)
            self.generator_samples.append(generator_sample)

            # Compute and accumulate losses
            self.compute_and_accumulate_loss(training_sample, latent_mean, latent_stddev, step)

        # Add tensorboard summary for the loss
        tf.summary.scalar("loss", self.loss)

        # Group all summaries into one variable we can keep hold of 
        self.merged_summary = tf.summary.merge_all()

        # Finally, make the train op (the optimizer)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)





    def create_recognition_network(self, step):
        """
        *Should only be called from within 'self.construct_network'*

        Creates the recognition network at some given step, say t. The network encodes q_phi(z_t | x), unless the 
        t > 0 *and* we are using x_t to predict z_t (in the "Latent InfoMax" version of the MC), in which case this 
        network encodes q_phi(z_t | x_t).

        Note the dependence on 'self.training_samples[step-1]'' and 'self.input_placeholder'

        In the "Latent InfoMax" version, we know x_t at both training and generation time, so we can sample from the 
        actual distribution. In all other versions, we don't know x at generation time, so we rely on regularization 
        to regularize the latent code to N(0,I), and we sample from N(0,I) instead.

        :param step: the time step with respect to the markov chain
        :return latent_mean: the mean for z_t given x_t
        :return latent_stddev: the stddev for z_t given x_t
        :return latent_train: tf variable to be used as the latent variable in the "training mode" of the network
        :return latent_generative: tf variable to be used as the latent variable in the "generative mode" of the network
        """
        latent_shape = tf.stack([tf.shape(self.input_placeholder)[0], self.latent_dim])

        if self.predict_latent_code and step != 0:
            latent_mean, latent_stddev = self.inference(self.training_samples[step-1], step)
            latent_sample = latent_mean + tf.multiply(latent_stddev, tf.random_normal(latent_shape))
            latent_train = latent_sample
            latent_generative = latent_sample
        else:
            latent_mean, latent_stddev = self.inference(self.input_placeholder, step)
            latent_sample = latent_mean + tf.multiply(latent_stddev, tf.random_normal(latent_shape))
            latent_train = latent_sample
            latent_generative = self.latents[-1]

        return latent_mean, latent_stddev, latent_train, latent_generative





    def create_recognition_network(self, last_training_sample, last_generative_sample, latent_train, latent_gen, step):
        """
        *Should only be called from within 'self.construct_network'*

        This creates the generator network, which itself consists of two parts. g_theta, an encoder for 
        'last_training_sample', and a decoder, which takes the encoded previous sample z'_t, and the latent state 
        sampled (either latent_train or latent_gen) z''_t, and uses them to generate a new sample x_t+1

        The encoding z'_t of x_t and the latent sample z''_t is combined within self.generator

        See self.generator_ladder for why we pass reuse=True to self.generator sometimes (we want the generator network 
        to be the same in both trianing and generative modes).

        :param last_training_sample: The last training sample, x_t, when run in training mode
        :param last_generative_sample: The last generative sample, x_t, when run in generative mode
        :param latent_train: The latent variable, z''_t, when run in training mode
        :param latent_gen: The latent variable, z''_t, when run in generative mode
        :param step: the time step with respect to the MC
        :return training_sample: the next training sample (for when running in training mode)
        :return generator_sample: the next generative sample (for when running int generative mode)
        """
        if step == 0:
            training_sample = self.generator(None, latent_train, step)
            generator_sample = self.generator(None, latent_gen, step, reuse=True)
        else:
            training_sample, resnet_ratios = self.generator(training_sample, latent_train, step)
            generator_sample, _ = self.generator(generator_sample, latent_placeholder, step, reuse=True)
            if self.add_debug_tb_variables:
                tf.summary.scalar("resnet_gate_weight_step_%d" % step, tf.reduce_mean(resnet_ratios))

        return training_sample, generator_sample







    def compute_and_accumulate_loss(self, training_sample, latent_mean, latent_stddev, step):
        """
        *Should only be called from within 'self.construct_network'*

        Compute the ELBO loss for the given MC step. This means that we want to compute the reconstruction loss 
        of the training sample vs the target_placeholder (== the input). We also want to compute the KL distance to 
        the unit normal.

        The function also accumulates losses in the variables self.loss

        If we are predicting the latent code (i.e. using q_phi(z_t | x_t) rather than q_phi(z_t | x) as a recognition 
        network), then we also wich to add the Latent InfoMax loss, defined in the paper (may be named something 
        other than "Latent InfoMax" later)

        :param training_sample: the current sample, when running in training mode, x_t
        :param latent_mean: The mean of the latent variable, z''_t, produced from the recognition network
        :param latent_stddev: The stddev's (per dim) of the latent variable z''_t, produced from the recognition network
        :param step: the time step with respect to the MC
        :return: None
        """

        batch_reconstruction_loss = tf.reduce_mean(tf.square(training_sample - self.target_placeholder), [1,2,3])
        batch_regularization_loss = tf.reduce_mean(-0.5 -tf.log(latent_stddev) +
                                            0.5 * tf.square(latent_stddev) +
                                            0.5 * tf.square(latent_mean), 1) 
        reconstruction_loss = tf.reduce_mean(batch_reconstruction_loss)
        regularization_loss = tf.reduce_mean(batch_regularization_loss)

        # Add to the overall loss
        if self.intermediate_reconstruction or step == self.mc_steps-1:
            self.loss += 16 * reconstruction_loss
        self.loss += self.reg_coeff * regularization_loss

        # If we're  we're running in 
        # Latent InfoMax mode. We want to add
        if self.predict_latent_code:
            # TODO update for the mattttthhhhh
            pass
            """
            mean = tf.reduce_mean(training_sample, 0)
            sample_var = tf.reduce_mean(tf.square(training_sample - mean))
            self.loss -= sample_var # want to MAXimize this variance, so we subtract it from the loss
            """

        # Keep track of the final reconstruction error (we just set this each time) 
        self.final_loss = tf.reduce_mean(reconstruction_loss)

        # Add tensorboards summaries for the losses at this step
        tf.summary.scalar("reconstruction_loss_step_%d" % step, reconstruction_loss)
        tf.summary.scalar("regularization_loss_step_%d" % step, regularization_loss)

        # Finally, prevent gradients from propogating between the different samples, if each encoder/decoder have their own losses
        if self.intermediate_reconstruction:
            training_sample = tf.stop_gradient(training_sample)





    def construct_network_homogeneous(self):
        """
        TODO
        """
        pass

        """ 
        TODO: this was the old "early stopping" code from the naive implemntation of the homogenous operation
            

            Enforcing the early stopping and computing the losses accordingly: 
            -----------------------------------------------------------------

            # Construct the loss for this step. (KL distance for regularizing the latent code and squared reconstruction loss for sample)
            # If early stopping is being used, first construct and apply a mask to the samples and losses
            if step != 0 and self.early_stopping_mc:
                training_improvements = tf.reduce_mean(tf.square(self.training_samples[step] - self.training_samples[step-1]), [1,2,3])
                training_energy = tf.reduce_mean(self.training_samples[step-1], [1,2,3]) # average pixel value from last iter (to check that we didn't cancel out last timestep). Called this energy for lack of a better term
                training_mask = threshold(tf.minimum(training_improvements, training_energy), self.early_stopping_threshold)
                tiled_training_mask = tf.tile(tf.reshape(training_mask, [-1,1,1,1]), tf.stack([1] + self.training_samples[step].get_shape().as_list()[1:])) # tile to broadcase how we want
                self.training_samples[step] = tf.multiply(tiled_training_mask, self.training_samples[step])

                generator_improvements = tf.reduce_mean(tf.square(self.generator_samples[step] - self.generator_samples[step-1]), [1,2,3])
                generator_energy = tf.reduce_mean(self.generator_samples[step-1], [1,2,3]) # average pixel value from last iter (to check that we didn't cancel out last timestep). Called this energy for lack of a better term
                generator_mask = threshold(tf.minimum(generator_improvements, generator_energy), self.early_stopping_threshold)
                tiled_generator_mask = tf.tile(tf.reshape(generator_mask, [-1,1,1,1]), tf.stack([1] + self.generator_samples[step].get_shape().as_list()[1:])) # tile to broadcast how we want
                self.generator_samples[step] = tf.multiply(tiled_generator_mask, self.generator_samples[step])

                batch_reconstruction_loss = tf.reduce_mean(tf.multiply(tiled_training_mask,
                                                    tf.square(training_sample - self.target_placeholder)), [1,2,3])
                tiled_training_mask = tf.tile(tf.reshape(training_mask, [-1, 1]), tf.stack([1] + latent_stddev.get_shape().as_list()[1:]))
                batch_regularization_loss = tf.reduce_mean(tf.multiply(tiled_training_mask,
                                                    -0.5 -tf.log(latent_stddev) +
                                                    0.5 * tf.square(latent_stddev) +
                                                    0.5 * tf.square(latent_mean)), 1)





            Computing the "final loss":
            --------------------------

            # Keep track of the final reconstruction error (we just set this each time) 
            # N.B. val = oldval + mask*(newval - oldval) is the same as "if mask == 1, val = newval"
            if step != 0 and self.early_stopping_mc:
                self.batch_final_loss = self.batch_final_loss * training_mask * (reconstruction_loss - self.batch_final_loss)
            else:
                self.batch_final_loss = reconstruction_loss
            self.final_loss = tf.reduce_mean(self.batch_final_loss) 
        """





    def log_tf_variables(self):
        """
        Log all tensorflow variables, useful for debugging sometimes!
        """
        self.LOG.debug("Printing out all variable names constructed (for debugging).")
        tf_vars = tf.trainable_variables()
        i = 0
        for var in tf_vars:
            self.LOG.debug("(%dth variable) %s" % (i, var.name))
            i += 1





    def train(self, input_batch, batch_target):
        """
        Perofrm ONE training update. (This is called from the main training loop in trainer.py)
        Also this will 

        :param input_batch: The input to the network for training
        :param batch_target: The target for the network (batch_target = input_batch - (noise if any))
        :return: The reconstruction loss of the FINAL sample (i.e. how well did the network do this time?)
        """
        self.iteration += 1

        # run values through our session 
        feed_dict = {self.input_placeholder: input_batch,
                     self.reg_coeff: 1 - math.exp(-self.iteration / self.reg_coeff_rate),
                     self.target_placeholder: batch_target}
        _, _, final_reconstruction_loss = self.sess.run([self.train_op, self.loss, self.final_loss], feed_dict=feed_dict)

        # Occasionally save the network params + write summaries to tensorboard
        if self.iteration % self.save_freq == 0:
            self.save_network()
        if self.iteration % self.tb_summary_freq == 0:
            summary = self.sess.run(self.merged_summary, feed_dict=feed_dict)
            self.writer.add_summary(summary, self.iteration)

        # return the final reconstruction loss (averaged per pixel)
        return final_reconstruction_loss / self.data_dims[0] / self.data_dims[1]





    def test(self, input_batch):
        """
        Runs the network, in training mode, on the test set. Used to test how we are doing.

        :param input_batch: The input to the network, noise that we want to turn into nice images
        :return: The final output of the SeqVAE network, i.e. the generated image
        """
        feed_dict = {self.input_placeholder: input_batch}
        generated_image = self.sess.run(self.training_samples[-1], feed_dict=feed_dict)
        return generated_image





    def generate_mc_samples(self, input_batch, batch_size=None):
        """
        Run the network in a generative mede. Generate sample from a normal distribution and feed 
        them into the network (as the latent variables). To run the generative part(s) of the network
        we run the self.generator_samples tf ops.

        N.B. This isn't used for optimization, just for visualization.

        :param input_batch: the input to the network (random noise) (None => use self.batch_size)
        :param batch_size: the size of the back
        :return: The generated samples from running the network in generative mode
        """
        if batch_size is None:
            batch_size = self.batch_size

        feed_dict = dict()
        feed_dict[self.input_placeholder] = input_batch
        for i in range(self.mc_steps):
            feed_dict[self.latents[i]] = np.random.normal(size=(batch_size, self.latent_dim))

        output = self.sess.run(self.generator_samples, feed_dict=feed_dict)
        return output





    def training_mc_samples(self, input_batch):
        """ 
        Run the network to generate samples in the training mode. To run the training part(s) 
        of the network we run the self.training_samples tf ops.

        N.B. This isn't used for optimization, just for visualization.

        :param input_batch: the input to the network (samples from the training set)
        :return: The generated samples from running the network in training mode
        """
        feed_dict = dict()
        feed_dict[self.input_placeholder] = input_batch
        output = self.sess.run(self.training_samples, feed_dict=feed_dict)
        return output





    def visualize(self, epoch, batch_size=10, use_gui=True):
        """
        Todo: Clean this function up + comment properly.

        Creates a visualization of the markov chain. 
        (Calls generate_mc_samples and training_mc_samples).
        """
        if use_gui is True and self.mc_fig is None:
            self.mc_fig, self.mc_ax = plt.subplots(1, 2)

        for i in range(2):
            if i == 0:
                bx = self.dataset.next_batch(batch_size)
                z = self.generate_mc_samples(bx, batch_size)
            else:
                bx = self.dataset.next_batch(batch_size)
                z = self.training_mc_samples(bx)
                z = [bx] + z
            v = np.zeros([z[0].shape[0] * self.data_dims[0], len(z) * self.data_dims[1], self.data_dims[2]])
            for b in range(0, z[0].shape[0]):
                for t in range(0, len(z)):
                    v[b*self.data_dims[0]:(b+1)*self.data_dims[0], t*self.data_dims[1]:(t+1)*self.data_dims[1]] 
                                                                                        = self.dataset.display(z[t][b])

            if use_gui is True:
                self.mc_ax[i].cla()
                if self.data_dims[-1] == 1:
                    self.mc_ax[i].imshow(v[:, :, 0], cmap='gray')
                else:
                    self.mc_ax[i].imshow(v)
                self.mc_ax[i].xaxis.set_visible(False)
                self.mc_ax[i].yaxis.set_visible(False)
                if i == 0:
                    self.mc_ax[i].set_title("test")
                else:
                    self.mc_ax[i].set_title("train")

            folder_name = 'models/%s/samples' % self.name
            if not os.path.isdir(folder_name):
                os.makedirs(folder_name)

            if v.shape[-1] == 1:
                v = v[:, :, 0]

            if i == 0:
                misc.imsave(os.path.join(folder_name, 'test_epoch%d.png' % epoch), v)
                misc.imsave(os.path.join(folder_name, 'test_current.png'), v)
            else:
                misc.imsave(os.path.join(folder_name, 'train_epoch%d.png' % epoch), v)
                misc.imsave(os.path.join(folder_name, 'train_current.png'), v)
        if use_gui is True:
            plt.draw()
            plt.pause(0.01)





    def inference_ladder(self, input_batch, step, reuse=False):
        """
        x -> z_t 
        or x_t -> z_t, if self.predict_latent_code is true

        Recognition network part of the VLAE. We can think of this as a heirarchical 
        encoder, so that we get heirarchical encodings of the input. Each level of this 
        heirarchical encoder is used to construct part of the latent space. And the latent 
        spaces from all the levels is concatenated at the end.

        N.B. This is the recognition network for one step in the MC. Variables are shared if 
        'self.share_recognition_params' is set to true.

        So this method will construct a network with ('self.vlae_levels' * 2) convolutional 
        "levels". For the ith heirarchical encodings/level we produce a latent state of dimension
        'self.vlae_latent_dims[i]'. Overall this network acts as a "recognition network", 
        in the contect of the original VAE paper.

        We note that at each level, we currently have two convolutions. The first has a stride 
        of two and halves the "image_size" (spatial dimensions) of the encodings. The second 
        has stride one and preserves image_size. Currently, we ignore 'self.image_sizes', but 
        we output a error and kill the program if they're not equal (because then the generator 
        ladder will not work)

        Latent variables take the form of multivariate gaussians (independent in each dimension),
        so to represent it, we simply output a mean and std_dev for each dimension

        :param input_batch: the input to encode (in the math this is x)
        :param step: the step in the overall markov chain (used for variable scoping)
        :param reuse: if we should reuse variables (n.b. we want the same variables for the training and generative 
                versions of the generative network, so sometimes this needs to be true, even in the inhomogeneous case)
        :return: the mean(s) and std_dev(s) of the latent state
        """
        if self.share_phi_weights:
            scope_name = "phi/inference_network"
            reuse = tf.AUTO_REUSE
        else:
            scope_name = "phi/inference_step_%d" % step

        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            cur_encoding = input_batch
            image_sizes = [cur_encoding.get_shape().as_list()[1]]
            latent_mean = []
            latent_stddev = []

            for level in range(self.vlae_levels-1):
                # encoding steps (move to next level in ladder)
                hidden = conv2d_bn_lrelu(cur_encoding, self.filter_sizes[level+1], [4,4], 2)
                cur_encoding = conv2d_bn_lrelu(hidden, self.filter_sizes[level+1], [4,4], 1)

                # latent code at this level in the ladder
                ladder = tf.reshape(cur_encoding, [-1, np.prod(cur_encoding.get_shape().as_list()[1:])])
                ladder_mean = layers.fully_connected(ladder, self.vlae_latent_dims[level], activation_fn=tf.identity)
                ladder_stddev = layers.fully_connected(ladder, self.vlae_latent_dims[level], activation_fn=tf.sigmoid)

                # maintain lists variables
                image_sizes.append(cur_encoding.get_shape().as_list()[1])
                latent_mean.append(ladder_mean)
                latent_stddev.append(ladder_stddev)

            # Add the last level (only have one convolution and a fully connected layer)
            cur_encoding = conv2d_bn_lrelu(cur_encoding, self.filter_sizes[self.vlae_levels-1], [4,4], 2)
            image_size = cur_encoding.get_shape().as_list()[1]
            cur_encoding = tf.reshape(cur_encoding, [-1, np.prod(cur_encoding.get_shape().as_list()[1:])])
            cur_encoding = fc_bn_lrelu(cur_encoding, self.filter_sizes[self.vlae_levels])

            ladder_mean = layers.fully_connected(ladder, self.vlae_latent_dims[self.vlae_levels-1], activation_fn=tf.identity)
            ladder_stddev = layers.fully_connected(ladder, self.vlae_latent_dims[self.vlae_levels-1], activation_fn=tf.sigmoid)

            image_sizes.append(image_size)
            latent_mean.append(ladder_mean)
            latent_stddev.append(ladder_stddev)

            # Check that 'self.image_sizes' is actually correct. If not, log an error and quit if it's not as 
            # the generative network will not work
            if len(image_sizes) != len(self.image_sizes):
                self.LOG.error("self.image_sizes is of length %d, but, image_sizes actually of length %d. " +
                    "(need len(self.image_sizes) == self.vlae_levels+1" % (len(image_sizes, len(self.image_sizes))))
                exit(-1)

            for i in range(len(image_sizes)):
                if image_sizes[i] != self.image_sizes[i]:
                    self.LOG.error("self.image_sizes and image_sizes in inference/generative networks don't match:")
                    self.LOG.error("self.image_sizes = %s" % self.image_sizes)
                    self.LOG.error("image_sizes = %s" % image_sizes)
                    exit(-1)

            # Return the complete latent space (concatenated means adn stddevs)
            return tf.concat(latent_mean, 1), tf.concat(latent_stddev, 1)





    def generator_ladder(self, input_batch, latent, step, reuse=False):
        """
        x_t-1, z_t -> x_t

        **Important:**
        Generative network part of the VLAE. Basically the same architecture as the inference network but run in reverse 
        (i.e. using transposed convolutions rather than convolutions).

        **Important:**
        A diagram is REALLY helpful to understand all of this.

        Inference encodings are computed from the input (x_t-1, the output from the previous step) and we add shortcuts 
        between the ith level of the inference network and the ith level of the generative network 

        We add residual connections over the whole autoencoder

        The input to the lowest level (i.e. level == self.vlae_levels) is the latent state

        The input to all other levels is the output of the previous level, with the latent state being added using 
        'self.combine_noise'

        For all levels, if input_batch is not null, we add shortcuts. Meaning that we directly add the encoding at the 
        ith layer of the reconition netowork to the ith level of the generator network
    
        :param input_batch: the output from the previous step (none if this is the first step) 
        :param latent: latent variables, sampled from a unit gaussian
        :param step: the current step in the markov chain
        :param reuse: if we should reuse variables (n.b. we want the same variables for the training and generative 
                versions of the generative network, so sometimes this needs to be true, even in the inhomogeneous case)
        :return: the output sample(s) from the generative network, and the residual connection ratio
        """
        # Compute the encodings of the input, z'_t, if we can
        encodings = None
        if input_batch is not None:
            encodings = self.compute_encodings(input_batch, step, reuse)

        # variable scope setup for decoding/generating
        if self.share_theta_weights and step != 0: # network is different on step 0 (no input_batch)
            scope_name = "theta/generative_network"
            reuse = tf.AUTO_REUSE
        else:
            scope_name = "theta/generative_step_%d" % step

        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            # split up latent variable for each level of ladder
            ladder = self.split_latent(latent)

            # First level of generator network / last level of ladder (needs to be reshaped to be used by conv 
            # transpose). Also need to combine in the latent variable z''_t, or 'the noise'.
            if encodings is not None:
                cur_sample = encodings[self.vlae_levels]
                cur_sample = self.combine_noise(cur_sample, ladder[self.vlae_levels-1])
            else:                
                cur_sample = ladder[self.vlae_levels-1]

            conv_shape = [self.image_sizes[self.vlae_levels], 
                     self.image_sizes[self.vlae_levels], 
                     self.filter_sizes[self.vlae_levels]]
            cur_sample = fc_bn_lrelu(cur_sample, np.prod(conv_shape))
            cur_sample = tf.reshape(cur_sample, [-1] + conv_shape)
            
            # Middle layers of the network (each iteration deals with 1 intermediate image size)
            # 2 transposed convolutions, adding in a shortcut from the encoded input if we have it (i.e. adding z'_t)
            # Also need to combine in the latent variable z''_t, or 'the noise' at each level
            for level in range(self.vlae_levels-2, -1, -1):
                deconv = conv2d_t_bn(cur_sample, self.filter_sizes[level+1], [4,4], 2, )
                if encodings is not None:
                    deconv = deconv + encodings[level+1]
                deconv = tf.nn.relu(deconv)

                deconv = self.combine_noise(deconv, ladder[level])
                cur_sample = conv2d_t_bn_relu(deconv, self.filter_sizes[level+1], [4,4], 1)

            # Final layer to get the output
            output = conv2d_t(cur_sample, self.data_dims[-1], [4,4], 2, activation_fn=tf.sigmoid)
            output = (self.dataset.range[1] - self.dataset.range[0]) * output + self.dataset.range[0]

            # Return output, adding resnet connection over the whole generator network if we can
            # (encodings[0] = input to inf network)
            if encodings is not None:
                ratio = conv2d_t(cur_sample, 1, [4,4], 2, activation_fn=tf.sigmoid)
                ratio = tf.tile(ratio, (1,1,1,self.data_dims[-)1]))
                output = tf.multiply(ratio, output) + tf.multiply(1-ratio, encodings[0])
                return output, ratio
            else:
                return output





    def compute_encodings(self, input_batch, step, reuse=False):
        """
        Creates a network to compute the encoding of some sample(s). That is, this network encodes g_theta, and 
        produces z'_t from x_t.

        :param input_batch: the output from the previous step (none if this is the first step) 
        :param step: the current step in the markov chain
        :param reuse: if we should reuse variables (n.b. we want the same variables for the training and generative 
                versions of the generative network, so sometimes this needs to be true, even in the inhomogeneous case)
        :return: Encodings for this given input
        """
        # variable scope setup for encoding input
        if self.share_theta_weights: 
            scope_name = "theta/generative_encoder_network"
            reuse = tf.AUTO_REUSE
        else:
            scope_name = "theta/generative_encoder_step_%d" % step

        # compute encodings from the input_batch (follows same structure as inference_ladder)
        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            cur_encoding = input_batch
            encodings = [cur_encoding]
            for level in range(self.vlae_levels-1):
                hidden = conv2d_bn_lrelu(cur_encoding, self.filter_sizes[level+1], [4,4], 2)
                cur_encoding = conv2d_bn_lrelu(hidden, self.filter_sizes[level+1], [4,4], 1)
                encodings.append(cur_encoding)

            cur_encoding = conv2d_bn_lrelu(cur_encoding, self.filter_sizes[self.vlae_levels-1], [4,4], 2)
            cur_encoding = tf.reshape(cur_encoding, [-1, np.prod(cur_encoding.get_shape().as_list()[1:])])
            cur_encoding = fc_bn_lrelu(cur_encoding, self.filter_sizes[self.vlae_levels])
            encodings.append(cur_encoding)

            return encodings





    def split_latent(self, latent):
        """
        Given the latent variable 'latent', split it into multiple variables. In the recognition network, we 
        concatinated lots of latent variables from different levels of the ladder. Here, we do the reverse operation 
        and split the latent state up for each level in the ladder. 

        Moreover, we add a fully connected layer, used to project them to the desired shape, so they can be 
        appropriately 'combined' in to the convolutional part of the neural network

        :param latent: the latent variable
        :return: ladder, a list of latent variables (the input latent split up, and projected onto appropriate shapes)
        """
        # split up the latent variables
        ladder = tf.split(latent, self.vlae_latent_dims, 1)
        for i in range(self.vlae_levels):
            ladder[i] = tf.reshape(ladder[i], [-1, self.vlae_latent_dims[i]])

        # project them into the correct image sizes to be added/concatenated to the appropriate intermediate representations
        for i in range(self.vlae_levels-1):
            ladder_step_size = self.image_sizes[i+1] * self.image_sizes[i+1] * self.filter_sizes[i+1]
            ladder[i] = fc_bn_lrelu(ladder[i], ladder_step_size)
            ladder[i] = tf.reshape(ladder[i], [-1, self.image_sizes[i+1], self.image_sizes[i+1], self.filter_sizes[i+1]])
        ladder[self.vlae_levels-1] = fc_bn_lrelu(ladder[self.vlae_levels-1], self.filter_sizes[self.vlae_levels+1])
        ladder[self.vlae_levels-1] = tf.reshape(ladder[self.vlae_levels-1], [-1, self.filter_sizes[self.vlae_levels+1]])

        return ladder





    def combine_noise(self, latent, ladder_embedding, name="default"):
        """
        Combine noise (the latent vatriable) into the ladder embedding. That is, for some 
        example, we have an encoded state, and we want to add the latent state to it 
        in some way. (Used in the generator network).

        We specify the way to combine the noise using:
        self.combine_noise_method

        It should take one of the following values:
        'concat': Concatenate the noise onto the embeddings
        'add': directly add the noise and the embeddings
        'gated_add': multiply the noise by a (trainable) variable, and then add

        :param latent: The latent variable we want to "combine" in
        :param ladder_embedding: The embedding/encoding of the input. Want to combine noise into this encoded space
        :param name: Name for this step, so that we can add tf/tensorboard summaries 
        :return: tensor having combined the embedding and latent "noise"
        """
        if self.combine_noise_method == 'concat':
            return tf.concat([latent, ladder_embedding], len(latent.get_shape()) - 1)

        elif self.combine_noise_method == 'add':
            return latent + ladder_embedding

        elif self.combine_noise_method == 'gated_add':
            gate = tf.get_variable("gate", shape=ladder.get_shape()[1:], initializer=tf.constant_initializer(0.1))
            tf.histogram_summary(name + "_noise_gate", gate)
            return latent + tf.multiply(gate, ladder_embedding)

