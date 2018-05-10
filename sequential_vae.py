from abstract_network import *
from scipy import misc
layers = tf.contrib.layers


def step(x):
    # x: tf tensor
    # return 0 if x < 0, return 1 if x >= 0
    return (tf.sign(x) + 1.0) / 2.0

def threshold(x, eps):
    # x: tf tensor
    # eps: float threshold
    # return 1 if x >= eps, which is true iff x/eps - 1 >= 0, return 0 otherwise
    return step(x/eps - 1.0)

def clip_grad_if_not_none(grad, clip_val):
    # grad: tf gradient tensor
    # clip_val: the value to clip by
    # returns tf.clip_by_value(grad, -self.clip_grad_value, self.clip_grad_value) if grad not None
    # N.B. tf.clip_by_value cannot handle None values...
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -clip_val, clip_val)


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


    We also have another mode of operation, which is a prototype/for debugging, and temporarily calling 
    "Infusion with Flat Convolutions Test". For details of how to run this and the (graphical) model, see the comments 
    for 'self.generator_flat'.
    """
    def __init__(self, dataset, batch_size, name, logger, version, base_dir):
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
        self.inference_first_step = the inference network to use on the first step (see self.inference too)
        self.generator_first_step = the generator network to use on the first step (see self.inference too)
        self.inference = the recognition network to use (currently just VLAE's recognition network is an option)
                    N.B. self.inference is a functional value
        self.generator = the generator network to use (currently just VLAE's generator is an option)
                    N.B. self.generator is a functional value
        self.mc_steps = the number of steps to use in the markov chain
        self.latent_dim = the dimension of the latent spaces (i.e. the number of dimensions z_i has for EACH i)
        self.intermediate_reconstruction = if each x_i should be a reconstruction of x. (If we should add a 
                    reconstruction loss for each x_i)
        self.early_stopping_mc = true, if when running the network in GENERATIVE mode, we stop the chain early if 
                    successive samples don't make enough improvement
        self.early_stopping_threshold = threshold on the L2 distance of successive samples for which we stop the chain,
                    note that we only stop early if 'self.early_stopping_mc' is true

        self.regularized_steps = the steps for which we should actually add a regulaizat
        self.predict_latent_code = true if we want to run the "Latent InfoMax" version of the MC. Can be used with either 
                    homogeneous or inhomeogeneous operation. This predicts the latent code using x_{t-1} rather than x 
                    (in BOTH gnerative and training samples).
        self.first_step_loss_coeff = the weighting on the first step of the chain (the math suggests this should 
                    be 2.0, and not 1, surprisingly)
        self.add_improvement_maximization_loss = if we should actually add the loss for the latent info max
        self.latent_mean_clip = a value to clip (per dimension) the latent mean predictions, set to inf by default to 
                    not provide any clipping
        self.latent_prior_stddev = the stddev on the prior that we use for the latent space (set to 1.0 for default)
        self.use_uniform_prior = if we want to use a uniform prior (but Gaussian estimation)
        self.add_noise_to_chain = if we want to add noise to samples in the chain. (N.B. This is technically what we 
                    should do, as the 'decoder' or 'generative' model outputs a mean of a (diagonal) Gaussian 
                    distribution). Outputting the mean is correct in a regular vae (as the Max Likelihood estimate), 
                    however, when used in a chain, we should actually sample the variable).
        self.predict_generator_noise = should we predeict the stddev of the Gaussian output by a generator network
        self.predict_generator_noise_as_scalar = if we should predict a scalar stddev for the samples (or if it should 
                    be a diagonal variance).
        self.predict_generator_stddev_max = max output from predicting the stddev of the noise 
        self.predict_generator_stddev_conv_layers = number of conv layers in stddev prediction
        self.predict_generator_stddev_filter_sizes = number of filters for each conv layer in the stddev prediction network 
        self.noise_stddevs = an array which must be exactly of lenght 'self.mc_steps', defining the std_dev for the 
                    Gaussian noise added at each step, if 'self.add_noise_to_chain' is true. (We still want to output 
                    the MLE estimate from the whole chain, so we should set self.noise_stddevs[-1] == 0.0).
        self.combine_noise_method = 'concat'/'add'/'gated_add', and specifies how to add in (latent) noise into the 
                    embeddings of the autoencoder (see combined_noise for more detail)

        'self.combine_noise_method' should take one of the following values:
        "concat": Concatenate the noise onto the embeddings
        "add": directly add the noise and the embeddings
        "gated_add": multiply the noise by a (trainable) variable, and then add

        (Flat Infusion Test)
        self.flat_conv_layers = the number of convolutional layers to use in the 
        self.flat_conv_filter_sizes = filter sizes for the "flat convolutions" part of the network. (That is, the 
                    filter sizes for each of the step in the markov chain). This should be a list exactly of length
                    'self.flat_conv_layers' + 1.


        (Hyperparams)
        self.learning_rate = the learning rate to use during training
        self.reg_coeff_rate = decides the rate of which we increase the (coefficient of) regularization on latent 
                    variables from 0 to 1 over time. (It's an inverse exponential, 1-e^-t)
        self.latent_pred_loss_coeff = the coefficient to the loss used in the "Latent InfoMax" version of the MC

        (Training params)
        self.save_freq = how frequently to save the network during training
        self.tb_summary_freq = how frequently to write summaries to tensorboard
        self.add_debug_tb_variables = if we should add tensorboard variables which are only necessary for debugging
        self.clip_grads = should we clip gradients
        self.clip_grad_value = what value should the gradient be clipped to?

        :param dataset: the dataset that we will be training this network on 
                        (implicitly defining the p_data we wish to model)
        :param batch_size: the batch size to use during training
        :param name: the name we are allocating this instand of the seqvae network 
        :param logger: a logger object, for logging 
        :param version: the version of this network that we are training
        :param base_dir: the base directory to use for saving any results
        :return: None
        """
        Network.__init__(self, dataset, logger, version, base_dir, name)

        self.dataset = dataset
        self.batch_size = batch_size
        self.data_dims = dataset.data_dims
        self.LOG = logger

        # VLAE parameters - assumes input image is square and at least a multiple of 16 (usually power of 2)
        self.vlae_levels = 4
        self.vlae_latent_dims = [3, 3, 3, 3]
        self.image_sizes = [self.data_dims[0], self.data_dims[0] // 2, 
                            self.data_dims[0] // 4, self.data_dims[0] // 8,
                            self.data_dims[0] // 16]
        self.filter_sizes = [self.data_dims[-1], 32, 64, 128, 384, 512]

        # SeqVAE parameters 
        self.share_theta_weights = False
        self.share_phi_weights = False
        self.inference_first_step = self.inference_ladder 
        self.generator_first_step = self.generator_ladder 
        self.inference = self.inference_ladder 
        self.generator = self.generator_ladder 
        self.mc_steps = 8
        self.latent_dim = np.sum(self.vlae_latent_dims)
        self.intermediate_reconstruction = True
        self.early_stopping_mc = False
        self.early_stopping_threshold = 0.0000000001
        self.regularized_steps = range(self.mc_steps)
        self.predict_latent_code = False
        self.first_step_loss_coeff = 1.0
        self.add_improvement_maximization_loss = False
        self.latent_mean_clip = np.inf
        self.predict_latent_code_with_regularization = False    # TEMPORARY VARIABLE FOR SOME TESTS
        self.latent_prior_stddev = 1.0
        self.use_uniform_prior = False 
        self.add_noise_to_chain = False
        self.predict_generator_noise = False
        self.predict_generator_noise_as_scalar = False
        self.predict_generator_stddev_max = 1.0
        self.predict_generator_stddev_conv_layers = 5 # with 5x5 convs, this gives a receptive field of 1+4*5 = 21
        self.predict_generator_stddev_filter_sizes  = [5, 5, 5, 5, 5]
        self.noise_stddevs = [0.5 ** 1, 0.5 ** 2, 0.5 ** 3, 0.5 ** 4, 0.5 ** 5, 0.5 ** 6, 0.5 ** 7, 0]
        self.combine_noise_method = "concat"

        # "Flat Conv Infusion" parameters
        self.flat_conv_layers = 6
        self.flat_conv_filter_sizes = [self.data_dims[-1], self.data_dims[-1]*2, self.data_dims[-1]*4, self.data_dims[-1]*8,
                                       self.data_dims[-1]*4, self.data_dims[-1]*2, self.data_dims[-1]]

        # Hyperparams and training params
        self.learning_rate = 0.0002
        self.reg_coeff_rate = 5000.0 # 1 epoch = 1000
        self.latent_pred_loss_coeff = 0.001 # 0.5, and reduce mean did really well, but very slow...
        self.save_freq = 2000
        self.tb_summary_freq = 10
        self.add_debug_tb_variables = True
        self.clip_grads = True
        self.clip_grad_value = 10.0


        # Config for different netnames, where customization is needed.
        # add overides for any of the above parameters here

        ####
        # 1: c_v2_diag_noise_abl
        # 2: c_homog_no_reg_imp_max
        # 3: c_v2_scalar_noise_abl
        # 4: c_v2_coeff_change_abl
        ####

        #####################
        # Single step VLAEs #
        #####################
        if self.name == "c_homog_one_step": #v2
            self.vlae_latent_dims = [12, 12, 12, 12]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 32, 64, 128, 384, 512]
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 1

        ##TODO##
        elif self.name == "m_homog_one_step":
            self.vlae_levels = 4
            self.vlae_latent_dims = [8, 8, 8, 8]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 1

        elif self.name == "s_homog_one_step": # v1
            self.vlae_latent_dims = [12, 12, 12, 12]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 32, 64, 128, 384, 512]
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 1

        ##TODO##
        elif self.name == "l_homog_one_step":
            pass

        ##############
        # V1 - Homog #
        ##############
        ##TODO##
        elif self.name == "c_homog_v1":
            self.vlae_latent_dims = [12, 12, 12, 12]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 16, 32, 64, 128, 384]
            self.share_theta_weights = True
            self.share_phi_weights = True
  
        ##TODO##          
        elif self.name == "m_homog_v1":
            pass

        ##TODO##            
        elif self.name == "s_homog_v1":
            pass
 
        ##TODO##           
        elif self.name == "l_homog_v1":
            pass



        ################
        # V1 - Inhomog #
        ################
        ##TODO##
        elif self.name == "c_inhomog_v1":
            self.vlae_latent_dims = [12, 12, 12, 12]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 16, 32, 64, 128, 384]


        ##TODO##            
        elif self.name == "m_inhomog_v1":
            pass
    
        ##TODO##        
        elif self.name == "s_inhomog_v1":
            pass
  
        ##TODO##          
        elif self.name == "l_inhomog_v1":
            pass



        ####################################
        # Ablations (v2) (just celeba?)    #
        # - latent regularization vs none  #
        # - coeff for log likelihood of x1 #
        # - improvement maximization loss  #
        # - sampling through the chain     #
        ####################################
        # regularized (this is v1, but with a new dependency model)
        elif self.name == "c_homog_reg_pred_latent": #v3 # "c_v2_reg"
            self.vlae_latent_dims = [12, 12, 12, 12]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 16, 32, 64, 128, 384]
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 8
            self.predict_latent_code = True
            self.latent_mean_clip = 32.0
            self.predict_latent_code_with_regularization = True
            

        # no regularization
        elif self.name == "c_homog_no_reg_pred_latent": # v5 # "c_v2_no_reg_abl
            self.vlae_latent_dims = [12, 12, 12, 12]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 16, 32, 64, 128, 384]
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 8
            self.predict_latent_code = True
            self.regularized_steps = [0]
            self.latent_mean_clip = 32.0


        # no reg + changed coeff regularization
        elif self.name == "c_v2_coeff_change_abl": 
            self.vlae_latent_dims = [12, 12, 12, 12]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 16, 32, 64, 128, 384]
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 8
            self.predict_latent_code = True
            self.regularized_steps = [0]
            self.latent_mean_clip = 32.0
            self.first_step_loss_coeff = 2.0

        # no reg + improvement max
        elif self.name == "c_homog_no_reg_imp_max": #v2 #"c_v2_imp_max_abl"
            self.vlae_latent_dims = [12, 12, 12, 12]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 16, 32, 64, 128, 384]
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 8
            self.predict_latent_code = True
            self.regularized_steps = [0]
            self.add_improvement_maximization_loss = True
            self.latent_mean_clip = 32.0


        ##TODO##
        # no reg + improvement max + changed coeff reg
        elif self.name == "c_v2_coeff_change_and_imp_max_abl":
            pass

        # no reg + improvement max + diagonal noise
        elif self.name == "c_v2_diag_noise_abl": # v2
            self.vlae_latent_dims = [12, 12, 12, 12]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 16, 32, 64, 128, 384]
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 8
            self.predict_latent_code = True
            self.regularized_steps = [0]
            self.latent_mean_clip = 32.0
            self.add_noise_to_chain = True
            self.predict_generator_noise = True
            self.predict_generator_stddev_max = 1.0


        # no reg + improvement max + scalar noise
        elif self.name == "c_v2_scalar_noise_abl": # v1
            self.vlae_latent_dims = [12, 12, 12, 12]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 16, 32, 64, 128, 384]
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 8
            self.predict_latent_code = True
            self.regularized_steps = [0]
            self.latent_mean_clip = 32.0
            self.add_noise_to_chain = True
            self.predict_generator_noise = True
            self.predict_generator_stddev_max = 1.0
            self.predict_generator_noise_as_scalar = True




        ###############################
        # Best model one all datasets #
        ###############################
        ##TODO##
        elif self.name == "c_final":
            pass

        ##TODO##
        elif self.name == "m_final":
            pass

        ##TODO##
        elif self.name == "s_final":
            pass

        ##TODO##
        elif self.name == "l_final":
            pass



        #######
        # Old #
        #######

        elif self.name == "c_homog_imp_max_var_pred":
            self.vlae_latent_dims = [12, 12, 12, 12]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 16, 32, 64, 128, 384]
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 8
            self.predict_latent_code = True
            self.add_improvement_maximization_loss = True
            self.latent_mean_clip = 32.0
            self.predict_latent_code_with_regularization = True
            self.add_noise_to_chain = True
            self.predict_generator_noise = True

        elif self.name == "c_homog_no_reg_pred_latent_var_pred":
            self.vlae_latent_dims = [12, 12, 12, 12]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 16, 32, 64, 128, 384]
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 8
            self.predict_latent_code = True
            self.regularized_steps = [0]
            self.latent_mean_clip = 32.0
            self.add_noise_to_chain = True
            self.predict_generator_noise = True

        elif self.name == "c_homog_no_reg_pred_latent_var_pred_more_noise":
            self.vlae_latent_dims = [12, 12, 12, 12]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 16, 32, 64, 128, 384]
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 8
            self.predict_latent_code = True
            self.regularized_steps = [0]
            self.latent_mean_clip = 32.0
            self.add_noise_to_chain = True
            self.predict_generator_noise = True
            self.predict_generator_stddev_max = 1.0

        # three
        elif self.name == "s_homog_imp_max":
            self.vlae_latent_dims = [12, 12, 12, 12]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 32, 64, 128, 384, 512]
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 8
            self.predict_latent_code = True
            self.add_improvement_maximization_loss = True
            self.latent_mean_clip = 32.0
            self.predict_latent_code_with_regularization = True

        # four
        elif self.name == "l_homog_imp_max":
            self.vlae_latent_dims = [12, 12, 12, 12]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 32, 64, 128, 384, 512]
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 8
            self.predict_latent_code = True
            self.add_improvement_maximization_loss = True
            self.latent_mean_clip = 32.0
            self.predict_latent_code_with_regularization = True

        # six
        elif self.name == "c_homog_imp_max":
            self.vlae_latent_dims = [12, 12, 12, 12]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.filter_sizes = [self.data_dims[-1], 16, 32, 64, 128, 384]
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 8
            self.predict_latent_code = True
            self.add_improvement_maximization_loss = True
            self.latent_mean_clip = 32.0
            self.predict_latent_code_with_regularization = True


        # five + seven
        elif self.name == "m_homog_imp_max_var_pred":
            self.vlae_levels = 3
            self.vlae_latent_dims = [8, 8, 8]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]

            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 6
            self.predict_latent_code = True
            self.add_improvement_maximization_loss = True
            self.latent_mean_clip = 32.0
            self.predict_latent_code_with_regularization = True
            self.add_noise_to_chain = True
            self.predict_generator_noise = True



        elif self.name == "sequential_vae_celebA_inhomog":
            # nothing
            pass

        elif self.name == "sequential_vae_celebA_homog":
            self.share_theta_weights = True
            self.share_phi_weights = True

        elif self.name == "sequential_vae_celebA_homog_early_stopping":
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.early_stopping_mc = True
            self.mc_steps = 15

        elif self.name == "sequential_vae_celebA_inhomog_inf_max":
            self.predict_latent_code = True

        elif self.name == "sequential_vae_celebA_homog_inf_max":
            self.share_theta_weights = True 
            self.share_phi_weights = True
            self.predict_latent_code = True

        elif self.name == "sequential_vae_celebA_homog_inf_max_early_stopping":
            self.share_theta_weights = True 
            self.share_phi_weights = True
            self.predict_latent_code = True
            self.mc_steps = 15
            self.early_stopping_mc = True

        elif self.name == "sequential_vae_celebA_inhomog_inf_max_uniform":
            self.predict_latent_code = True
            self.use_uniform_prior = True

        elif self.name == "vlae_celebA":
            self.mc_steps = 1
            self.vlae_latent_dims = [16, 16, 16, 16] # same latent size as 8 stage markov chain
            self.latent_dim = np.sum(self.vlae_latent_dims)

        elif self.name == "sequential_vae_celebA_homog_fixed_length":
            self.share_theta_weights = True 

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

        elif self.name == "c_inhomog":
            pass

        elif self.name == "c_homog":
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.mc_steps = 25

        elif self.name == "c_early_stop":
            self.mc_steps = 15
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.early_stopping_mc = True

        elif self.name == "c_inhomog_inf_max":
            self.predict_latent_code = True
            self.latent_mean_clip = 8.0

        elif self.name == "c_homog_inf_max_clipped":
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.predict_latent_code = True
            self.latent_mean_clip = 8.0

        elif self.name == "c_inhomog_inf_max_clipped":
            self.predict_latent_code = True
            self.latent_mean_clip = 8.0

        elif self.name == "c_homog_inf_max_regularized":
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.predict_latent_code = True
            self.predict_latent_code_with_regularization = True

        elif self.name == "c_sample_images":
            self.add_noise_to_chain = True

        elif self.name == "c_homog_sample_images":
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.add_noise_to_chain = True

        elif self.name == "c_infusion_test":
            self.share_theta_weights = False
            self.share_phi_weights = False
            self.generator = self.generator_flat
            self.add_noise_to_chain = True

        elif self.name == "c_homog_infusion_test":
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.generator = self.generator_flat
            self.add_noise_to_chain = True
        
        elif self.name == "m_infusion_test":
            self.share_theta_weights = False
            self.share_phi_weights = False
            self.generator = self.generator_flat
            self.add_noise_to_chain = True
            self.vlae_levels = 3
            self.vlae_latent_dims = [2, 2, 2]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]
            self.mc_steps = 5

        elif self.name == "c_homog_infusion_test": 
            self.share_theta_weights = False
            self.share_phi_weights = False
            self.generator = self.generator_flat
            self.add_noise_to_chain = True
            self.share_theta_weights = True
            self.share_phi_weights = True


        elif self.name == "sequential_vae_mnist_homog_early_stopping":
            self.vlae_levels = 3
            self.vlae_latent_dims = [2, 2, 2]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]
            self.mc_steps = 15
            self.share_theta_weights = True
            self.share_phi_weights = True
            self.early_stopping_mc = True

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

        # python main.py --dataset=mnist --plot_reconstruction --netname=m_inhomog --version=1
        elif self.name == "m_inhomog":
            self.vlae_levels = 3
            self.vlae_latent_dims = [2, 2, 2]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]
            self.mc_steps = 5

        elif self.name == "m_noise":
            self.vlae_levels = 3
            self.vlae_latent_dims = [4, 4, 4]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]
            self.mc_steps = 5
            self.add_noise_to_chain = True
            self.noise_stddevs = [8.0, 4.0, 1.0, 0.25, 0]

        else:
            self.LOG.error("Unknown network name %s" % self.name)
            exit(-1)
    
        # Construct initialize and print network
        self.construct_network()
        self.init_network()
        self.print_network()
        self.log_tf_variables()

        # stuff for figures
        self.mc_fig, self.mc_ax = None, None





    def construct_network(self):
        """
        Construct the network, using 'self.inference' and 'self.generator'. This network will produce
        the sequence of samples and latent random variables.

        Tensorflow variables defined here (and the subsequent function calls this function makes):
        self.input_placeholder = placeholder for the input image (= the target image + optional noise)
        self.target_placeholder =  placeholder for the target image (= input image, without any noise)
        self.reg_coeff = placeholder (with default value) for the coefficient of regularization (slowly raised from zero
                    to one, to implement "warm starting"). We also use this as the coefficient for 
                    'self.latent_pred_loss', as that's a little unstable early on in training.
        
        self.latents = placeholders for latent variables, use these when we want to use the network generatively
        self.training_mles = the means/maximum likelihood estimates of samples along the chain. These are the means 
                    of the Diagonal Gaussian's used to generate the samples in the network running in the training mode
        self.training_samples = samples generated along the chain, use these one's when training the network this uses a 
                    latent variable that's sampled using the mean and stddev from inference network of the previous 
                    sample
        self.generative_mles = the means/maximum likelihood estimates of samples along the chain. These are the means 
                    of the Diagonal Gaussian's used to generate the samples in the network running in the generative mode
        self.generative_samples = samples generated along the chain, use these one's when running the generatively this 
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
        self.training_mles = []
        self.training_samples = []
        self.generative_mles = []
        self.generative_samples = []

        self.loss = 0.0
        self.improvement_maximization_loss = 0.0
        self.final_loss = None

        training_mle = None
        training_sample = None
        generative_mle = None
        generative_sample = None

        for step in range(self.mc_steps):
            prev_training_mle = training_mle
            prev_training_sample = training_sample
            prev_generative_sample = generative_sample

            # Create placeholder for latent variable on this step. i.e. z_i (used in "generative mode")
            latent_placeholder = tf.placeholder(shape = [None, self.latent_dim], 
                                                dtype = tf.float32, 
                                                name = ("latent_placeholder_%d" % step))
            self.latents.append(latent_placeholder)

            # On the first step, let x_0 be uniform random noise
            # And If we're adding noise through the chain, pass this noise into the first step of the generator network
            if step == 0:
                generative_sample = tf.random_uniform(shape=tf.stack([tf.shape(self.input_placeholder)[0]] + self.data_dims))
                self.generative_samples.append(generative_sample)
                if self. add_noise_to_chain:
                    prev_training_sample = generative_sample
                    prev_generative_sample = generative_sample


            # Make recognition, p_phi(z_t|x), and generative, p_theta(x_t|z_t,x_t-1), networks. Append samples from them
            # latent_train = the latent variable in training mode, latent_generative = in generative mode
            latent_mean, latent_stddevs, latent_train, latent_generative = self.create_recognition_network(step=step)
            generator_network_output = self.create_generator_network(prev_training_sample, prev_generative_sample, 
                                                                                latent_train, latent_generative, step)
            training_mle, training_stddevs, training_sample, generative_mle, generative_sample = generator_network_output

            self.training_mles.append(training_mle)
            self.training_samples.append(training_sample)
            self.generative_mles.append(generative_mle)
            self.generative_samples.append(generative_sample)

            # Add some tensor board variables for debugging what's going on
            if self.add_debug_tb_variables:
                tf.summary.scalar("training_stddev_avg_magnitude_step_%d" % step, tf.reduce_mean(tf.abs(training_stddevs)))
                tf.summary.scalar("latent_mean_avg_magnitude_step_%d" % step, tf.reduce_mean(tf.abs(latent_mean)))
                tf.summary.scalar("latent_mean_avg_step_%d" % step, tf.reduce_mean(latent_mean))
                tf.summary.scalar("latent_stddev_avg_step_%d" % step, tf.reduce_mean(latent_stddevs))

            # Compute and accumulate losses
            self.compute_and_accumulate_loss(prev_training_mle, training_mle, training_stddevs, latent_train, latent_mean, latent_stddevs, step)

        # Add tensorboard summary for the loss
        tf.summary.scalar("loss", self.loss)

        # Group all summaries into one variable we can keep hold of 
        self.merged_summary = tf.summary.merge_all()

        # Make the training op in self.train_op
        self.make_training_op()






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

        if self.predict_latent_code:
            if step != 0:
                latent_mean, latent_stddev = self.inference(self.training_samples[step-1], step)
            else:
                latent_mean, latent_stddev = self.inference_first_step(self.input_placeholder, step)
            latent_sample = latent_mean + tf.multiply(latent_stddev, tf.random_normal(latent_shape))
            latent_train = latent_sample
            latent_generative = latent_sample
        else:
            latent_mean, latent_stddev = self.inference(self.input_placeholder, step)
            latent_sample = latent_mean + tf.multiply(latent_stddev, tf.random_normal(latent_shape))
            latent_train = latent_sample
            latent_generative = self.latents[-1]

        return latent_mean, latent_stddev, latent_train, latent_generative





    def create_generator_network(self, last_training_sample, last_generative_sample, latent_train, latent_gen, step):
        """
        *Should only be called from within 'self.construct_network'*

        This creates the generator network, which itself consists of two parts. g_theta, an encoder for 
        'last_training_sample', and a decoder, which takes the encoded previous sample z'_t, and the latent state 
        sampled (either latent_train or latent_gen) z''_t, and uses them to generate a new sample x_t+1

        The encoding z'_t of x_t and the latent sample z''_t is combined within self.generator

        See self.generator_ladder for why we pass reuse=True to self.generator sometimes (we want the generator network 
        to be the same in both trianing and generative modes).

        If we have self.early_stopping_mc = True, then we need to identify if this step (when running in GENERATOR mode) 
        didn't make a significant enough improvement. If so, we return a zeroed 'generative_sample'

        We also add noise to the samples if 'self.add_noise_to_chain' is true, and the amount of noise is returned by 
        the generator networks. If self.predict_generator_noise is true, the the generator network will train and learn 
        to predict the stddev to use with the sample (i.e. it outputs how "confident" it is in the sample it produced).
        If 'self.add_noise_to_chain' is false, then self.generator should return the correct value from 
        self.noise_stddevs, to use a fixed noise with that 

        We also multiply the noise_stddevs by self.reg_coeff, so that noise is slowly learned, as we can think of the 
        noise as regularization, and it can also make training early on unstable

        :param last_training_sample: The last training sample, x_t, when run in training mode
        :param last_generative_sample: The last generative sample, x_t, when run in generative mode
        :param latent_train: The latent variable, z''_t, when run in training mode
        :param latent_gen: The latent variable, z''_t, when run in generative mode
        :param step: the time step with respect to the MC
        :return training_sample: the next training sample (for when running in training mode)
        :return generative_sample: the next generative sample (for when running int generative mode)
        """
        if step == 0:
            training_mle, training_stddevs, _ = self.generator_first_step(None, latent_train, step)
            generative_mle, generative_stddevs, _ = self.generator_first_step(None, latent_gen, step, reuse=True)
        else:
            training_mle, training_stddevs, resnet_ratios = self.generator(last_training_sample, latent_train, step)
            generative_mle, generative_stddevs, _ = self.generator(last_generative_sample, latent_gen, step, reuse=True)
            if self.add_debug_tb_variables and step is not None:
                tf.summary.scalar("resnet_gate_weight_step_%d" % step, tf.reduce_mean(resnet_ratios))

            # At test time/generative mode. If we either don't make any improvement this step, or, the chain was already 
            # cut off (last_generative_sample \approx 0), them we should zero out 'generative_sample'. We compute the 
            # change between the two images (L2 norm) to detect lack of change. Threshold is a helper defined at the 
            # beginning of the file. To zero examples in the batch out, we compute a mask and multiply by that.
            if self.early_stopping_mc:
                improvements = tf.sqrt(tf.reduce_sum(tf.square(generative_sample - last_generative_sample), [1,2,3]))
                last_avg_val =  tf.reduce_mean(last_generative_sample, [1,2,3]) 
                batch_mask = threshold(tf.minimum(improvements, last_avg_val), self.early_stopping_threshold)
                tiled_mask = tf.tile(tf.reshape(batch_mask, [-1,1,1,1]), tf.stack([1] + generative_sample.get_shape().as_list()[1:]))
                generative_sample = tf.multiply(tiled_mask, generative_sample)

        # Add any noise indicated by the generator network
        image_batch_shape = tf.stack([tf.shape(self.input_placeholder)[0]] + self.data_dims)
        training_sample = training_mle + self.reg_coeff  * training_stddevs * tf.random_normal(image_batch_shape)
        generative_sample = generative_mle + self.reg_coeff * generative_stddevs * tf.random_normal(image_batch_shape)

        return training_mle, training_stddevs, training_sample, generative_mle, generative_sample







    def compute_and_accumulate_loss(self, prev_training_mle, training_mle, training_stddevs, latent_sample, 
                                    latent_mean, latent_stddev, step):
        """
        *Should only be called from within 'self.construct_network'*

        Compute the ELBO loss for the given MC step. This means that we want to compute the reconstruction loss 
        of the training sample vs the target_placeholder (== the input). We also want to compute the KL distance to 
        the unit normal.

        If we are predicting the generator noise, then we need to include the stddevs in the reconstruction loss (as 
        we no longer assume that the stddev is 1/2).

        The function also accumulates losses in the variables self.loss

        The KL distance between N(mu, sigma), and N(0, sigma2) is:
        sum_i 1 + log(sigma_i) - log(sigma2_i) - sigma_i^2/2sigma2_i^2 - mu_i^2/2sigma2_i^2 
        where the index i is with respect to dimension.

        For us, sigma2 is fixed, so -log(sigma2) is a constant, and we ignore it in our loss function.

        If we are predicting the latent code (i.e. using q_phi(z_t | x_t) rather than q_phi(z_t | x) as a recognition 
        network), then we also wich to add the Latent InfoMax loss (see write up for math), defined in the paper (may 
        be named something other than "Latent InfoMax" later)

        :param prev_training_mle: the previous training mle (output from generator network) running in training mode. 
                    At time 'step' = 0, this should be None. Used only if we are running in Latent InfoMax mode, to 
                    compute the Latent InfoMax loss
        :param training_mle: the current mle, output from the generator network, when running in training mode, x_t
        :param training_stddevs: the stddevs output from the generator network for the sample, when run in training mode, 
                    this is the stddev of x_t
        :param latent_sample: a sample of the latent variable z_t, sampled from q_phi(z_t | x_t)
        :param latent_mean: The mean of the latent variable, z''_t, produced from the recognition network
        :param latent_stddev: The stddev's (per dim) of the latent variable z''_t, produced from the recognition network
        :param latent_samples_for_lp: Samples of the latent in the form of a tensor with shape [batch_size,
                    latent_dims, num_samples]. Used to estimate expectations for the loss in 'Latent Info Max' mode.
        :param latent_samples_for_latent_pred_loss: Samples of the latent in the form of a tensor with shape [batch_size, 
                    num_samples, latent_dim]. Used to estimate expectations for the loss in 'Latent Info Max' mode.
                    "for_lp" = "for latent prediction loss"
        :param img_samples_for_lp: Samples of the latent in the form of a tensor with shape [batch_size, 
                    num_samples, img_wifth, img_height, img_depth]. Used to estimate expectations for the loss in 
                    'Latent Info Max' mode. "for_lp" = "for latent prediction loss"
        :param step: the time step with respect to the MC
        :return: None
        """
        # reconstruction loss, add a tensorboard summary for the normal reconstruction loss when we're adding sigma predictions into the mix (to be able to compare multiple)
        batch_reconstruction_loss = tf.reduce_mean(tf.square(training_mle - self.target_placeholder), [1,2,3])
        if self.predict_generator_noise:
            tf.summary.scalar("classic_reconstruction_loss_step_%d" % step, tf.reduce_mean(batch_reconstruction_loss))
            batch_reconstruction_loss = tf.reduce_mean(tf.log(training_stddevs) + 0.5 * tf.log(2 * np.pi) +
                                                0.5 * tf.square((training_mle - self.target_placeholder) / training_stddevs))

        # add regularization loss (if we should)
        batch_regularization_loss = 0
        if step in self.regularized_steps:
            if not self.use_uniform_prior:
                batch_regularization_loss = tf.reduce_mean(-0.5 -tf.log(latent_stddev) +
                                                    0.5 * tf.square(latent_stddev) / (self.latent_prior_stddev ** 2) +
                                                    0.5 * tf.square(latent_mean) / (self.latent_prior_stddev ** 2), 1) 
            else:
                batch_regularization_loss = tf.reduce_mean(-tf.log(latent_stddev))

        # aggregate over batch - TODO: remove this... it's pointless, just take the mean properly above...
        reconstruction_loss = tf.reduce_mean(batch_reconstruction_loss)
        regularization_loss = tf.reduce_mean(batch_regularization_loss)

        # Add reconstruction loss to the overall loss
        if self.intermediate_reconstruction or step == self.mc_steps-1:
            self.loss += 16 * reconstruction_loss

        # Add regularization. Only want to add regularization in Latent InfoMax mode on the first step
        if not self.predict_latent_code or self.predict_latent_code_with_regularization or step == 0:
            self.loss += self.reg_coeff * regularization_loss

        # Scale the loss according to self.first_step_loss_coeff if it's the first step
        if step == 0:
            self.loss *= self.first_step_loss_coeff

        # If we're  we're running in Latent InfoMax mode. We want to add a loss to optimize the phi variables according 
        # to the new objective. Also make it "warm started", because it's a little unstable early on
        if self.add_improvement_maximization_loss and prev_training_mle is not None:
            # probs of latent vars
            normal_distr = tf.contrib.distributions.MultivariateNormalDiag(latent_mean, latent_stddev)
            latent_probs = tf.exp(normal_distr.log_prob(latent_sample))

            # Compute squared norm
            diffs = training_mle - prev_training_mle
            norms = tf.reduce_sum(diffs ** 2, axis=[1,2,3])

            # Weight by the probability to account for potentially bad samples
            weighted_norms = norms#latent_probs * norms

            # set loss to be negative, so we maximize
            latent_loss = self.reg_coeff * self.latent_pred_loss_coeff * -tf.reduce_mean(weighted_norms)
            self.improvement_maximization_loss += latent_loss
            tf.summary.scalar("improvement_maximization_loss_step_%d" % step, latent_loss)

        # Keep track of the final reconstruction error (we just set this each time) 
        self.final_loss = tf.reduce_mean(reconstruction_loss)

        # Add tensorboards summaries for the losses at this step
        tf.summary.scalar("reconstruction_loss_step_%d" % step, reconstruction_loss)
        tf.summary.scalar("regularization_loss_step_%d" % step, regularization_loss)

        # Finally, prevent gradients from propogating between the different samples, if each encoder/decoder have their own losses
        if self.intermediate_reconstruction:
            training_mle = tf.stop_gradient(training_mle)





    def get_subset_weights(self, substr):
        """
        Get a subset of the trainable weights, whos name contains 'substr' as a substring in it's name

        :param substr: the substring to search for in the weight names
        :return: the set of weights with substr as a substring of the variable name
        """
        tf_vars = tf.trainable_variables()
        subset = []
        for var in tf_vars:
            if substr in var.name:
                subset.append(var)
        return subset





    def get_all_weights(self):
        """
        To get all weights, use that "" is a substring of any other string
        """
        return self.get_subset_weights("")





    def make_training_op(self):
        """
        *Should only be called from within 'self.construct_network'*

        Computes self.train_op, the tf op used to train the network, which combines updates for the ELBO loss, and the 
        latent prediction loss, that were computed in 'self.compute_and_accumulate_loss'

        If we are predicting the latent code, only want to have the latent prediction variables (phi) try to maximize 
        the varience of the output (it would be silly to make this an objective of the generative model (theta))

        Also performs some processing on the gradients, including gradient clipping. Because tf is silly, if a grad is 
        zero, it returns a None tensor, AND, tf.clip_by_value cannot handle None values.... So we use the helper function 
        'clip_grad_if_not_none', to check for if the value is none to get around this problem

        And computes some useful sanity checks to watch in training if we're debugging, and also adds a lot of 
        tensorboard scalars for us to use
        """
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        all_vars = self.get_all_weights()
        theta_vars = self.get_subset_weights("theta")
        phi_vars = self.get_subset_weights("phi")

        grads = optimizer.compute_gradients(self.loss, var_list=all_vars)
        if self.clip_grads:
            grads = [(clip_grad_if_not_none(grad, self.clip_grad_value), var) for grad, var in grads]
        elbo_train_op = optimizer.apply_gradients(grads)

        if self.add_debug_tb_variables:
            theta_weights_norm = tf.global_norm(theta_vars)
            phi_weights_norm = tf.global_norm(phi_vars)

            theta_elbo_grads = [grad for grad, var in grads if ("theta" in var.name)]
            phi_elbo_grads = [grad for grad, var in grads if ("phi" in var.name)]

            theta_elbo_grads_norm = tf.global_norm(theta_elbo_grads)
            phi_elbo_grads_norm = tf.global_norm(phi_elbo_grads)

            theta_elbo_update_ratio = self.learning_rate * theta_elbo_grads_norm / theta_weights_norm
            phi_elbo_update_ratio = self.learning_rate * phi_elbo_grads_norm / phi_weights_norm

            tf.summary.scalar("theta_weights_norm", theta_weights_norm)
            tf.summary.scalar("phi_weights_norm", phi_weights_norm)
            tf.summary.scalar("theta_elbo_grads_norm", theta_elbo_grads_norm)
            tf.summary.scalar("phi_elbo_grads_norm", phi_elbo_grads_norm)
            tf.summary.scalar("theta_elbo_update_ratio", theta_elbo_update_ratio)
            tf.summary.scalar("phi_elbo_update_ratio", phi_elbo_update_ratio)


        if not self.add_improvement_maximization_loss:
            self.train_op = elbo_train_op

        else:
            grads = optimizer.compute_gradients(self.improvement_maximization_loss, var_list=phi_vars)
            if self.clip_grads:
                grads = [(clip_grad_if_not_none(grad, self.clip_grad_value), var) for grad, var in grads]
            pred_latent_train_op = optimizer.apply_gradients(grads)

            if self.add_debug_tb_variables:
                phi_latent_pred_grads_norm = tf.global_norm(list(zip(*grads))[0])
                phi_latent_pred_update_ratio = self.learning_rate * phi_latent_pred_grads_norm / phi_weights_norm

                tf.summary.scalar("phi_latent_pred_grads_norm", phi_latent_pred_grads_norm)
                tf.summary.scalar("phi_latent_pred_update_ratio", phi_latent_pred_update_ratio)

            self.train_op = tf.group(elbo_train_op, pred_latent_train_op)





    def log_tf_variables(self):
        """
        Log all tensorflow variables, useful for debugigng sometimes!
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
        Runs the network, in training mode, on the test set. Used to test how we are doing. (Returns the mle output from 
        the final network)

        :param input_batch: The input to the network, noise that we want to turn into nice images
        :return: The final output of the SeqVAE network, i.e. the generated image
        """
        feed_dict = {self.input_placeholder: input_batch}
        generated_mle, generated_image = self.sess.run([self.training_mles[-1], self.training_samples[-1]], feed_dict=feed_dict)
        return generated_mle





    def generate_mc_samples(self, input_batch, batch_size=None):
        """
        Run the network in a generative mede. Generate sample from a normal distribution and feed 
        them into the network (as the latent variables). To run the generative part(s) of the network
        we run the self.generative_samples tf ops.

        If we are sampling the x_t from a normal distribution properly, then we should also return the means/MLEs that 
        we used for sampling

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

        if self.add_noise_to_chain:
            output = self.sess.run(self.generative_mles + self.generative_samples, feed_dict=feed_dict)
        else:
            output = self.sess.run(self.generative_samples, feed_dict=feed_dict)

        return output





    def training_mc_samples(self, input_batch):
        """ 
        Run the network to generate samples in the training mode. To run the training part(s) 
        of the network we run the self.training_samples tf ops.

        If we are sampling the x_t from a normal distribution properly, then we should also return the means/MLEs that 
        we used for sampling

        N.B. This isn't used for optimization, just for visualization.

        :param input_batch: the input to the network (samples from the training set)
        :return: The generated samples from running the network in training mode
        """
        feed_dict = dict()
        feed_dict[self.input_placeholder] = input_batch

        if self.add_noise_to_chain:
            output = self.sess.run(self.training_mles + self.training_samples, feed_dict=feed_dict)
        else:
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

            if self.add_noise_to_chain:
                # z[0].shape[0] = batch size
                chain_len = (len(z)+1) // 2
                v = np.zeros([z[0].shape[0] * self.data_dims[0] * 2, chain_len * self.data_dims[1], self.data_dims[2]])
            else:
                v = np.zeros([z[0].shape[0] * self.data_dims[0], len(z) * self.data_dims[1], self.data_dims[2]])

            for b in range(0, z[0].shape[0]):
                for t in range(0, len(z)):
                    tc = t + (1-i)
                    if self.add_noise_to_chain and tc < chain_len:
                        v[2*b*self.data_dims[0]:(2*b+1)*self.data_dims[0], tc*self.data_dims[1]:(tc+1)*self.data_dims[1]] = \
                                                                                        self.dataset.display(z[t][b])
                    elif self.add_noise_to_chain: # and t >= chain_len:
                        u = tc - chain_len + i
                        v[(2*b+1)*self.data_dims[0]:(2*b+2)*self.data_dims[0], u*self.data_dims[1]:(u+1)*self.data_dims[1]] = \
                                                                                        self.dataset.display(z[t][b])
                    else:
                        v[b*self.data_dims[0]:(b+1)*self.data_dims[0], t*self.data_dims[1]:(t+1)*self.data_dims[1]] = \
                                                                                        self.dataset.display(z[t][b])

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

            folder_name = self.base_dir + "/samples"
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

        Recognition network part of the VLAE. We can think of this as a heirarchical encoder, so that we get 
        heirarchical encodings of the input. Each level of this heirarchical encoder is used to construct part of the 
        latent space. And the latent spaces from all the levels is concatenated at the end.

        N.B. This is the recognition network for one step in the MC. Variables are shared if 
        'self.share_recognition_params' is set to true.

        So this method will construct a network with ('self.vlae_levels' * 2) convolutional "levels". For the ith 
        heirarchical encodings/level we produce a latent state of dimension 'self.vlae_latent_dims[i]'. Overall this 
        network acts as a "recognition network", in the contect of the original VAE paper.

        We note that at each level, we currently have two convolutions. The first has a stride of two and halves the 
        "image_size" (spatial dimensions) of the encodings. The second has stride one and preserves image_size. 
        Currently, we ignore 'self.image_sizes', but we output a error and kill the program if they're not equal 
        (because then the generator ladder will not work)

        Latent variables take the form of multivariate gaussians (independent in each dimension), so to represent it, 
        we simply output a mean and std_dev for each dimension. Means can be clipped at a certain max magnitude, if we 
        wish.

        In Latent InfoMax mode, the first step recognition takes the training sample and should be normalized to a 
        unit gaussian. Otherwise, we don't do that. So in latent infomax mode 

        :param input_batch: the input to encode (in the math this is x)
        :param step: the step in the overall markov chain (used for variable scoping)
        :param reuse: if we should reuse variables (n.b. we want the same variables for the training and generative 
                versions of the generative network, so sometimes this needs to be true, even in the inhomogeneous case)
        :return: the mean(s) and std_dev(s) of the latent state
        """
        # network is different on step 0 (when input_batch = ground truth), vs a sample in info max mode. 
        # otherwise, all the networks are the same outside of info max mode
        if not self.share_phi_weights or (self.predict_latent_code and step == 0): 
            scope_name = "phi/inference_step_%d" % step
        else:
            scope_name = "phi/inference_network"
            reuse = tf.AUTO_REUSE

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
                ladder_mean = tf.clip_by_value(ladder_mean, -self.latent_mean_clip, self.latent_mean_clip)
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
            ladder_mean = tf.clip_by_value(ladder_mean, -self.latent_mean_clip, self.latent_mean_clip)
            ladder_stddev = layers.fully_connected(ladder, self.vlae_latent_dims[self.vlae_levels-1], activation_fn=tf.sigmoid)

            image_sizes.append(image_size)
            latent_mean.append(ladder_mean)
            latent_stddev.append(ladder_stddev)

            # Check that 'self.image_sizes' is actually correct. If not, log an error and quit if it's not as 
            # the generative network will not work
            if len(image_sizes) != len(self.image_sizes):
                self.LOG.error("self.image_sizes is of length %d, but, image_sizes actually of length %d. " +
                    "(need len(self.image_sizes) == self.vlae_levels+1" % (len(image_sizes), len(self.image_sizes)))
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

        **Important:**
        This canse build two different networks, one for the first step where we don't have an x_t-1, and the other is 
        when we do have an x_t-1. The differences can be seen with the conditional blocks "if encodings is not null".

        Inference encodings are computed from the input (x_t-1, the output from the previous step) and we add shortcuts 
        between the ith level of the inference network and the ith level of the generative network 

        We add residual/highway connections over the whole autoencoder (i.e. between two steps in the chain)

        The input to the lowest level (i.e. level == self.vlae_levels) is the latent state

        The input to all other levels is the output of the previous level, with the latent state being added using 
        'self.combine_noise'

        For all levels, if input_batch is not null, we add shortcuts. Meaning that we directly add the encoding at the 
        ith layer of the reconition netowork to the ith level of the generator network

        The generator network also needs to predict stddevs for the sample, which should work as follows:
        if self.add_noise_to_chain == False: stddev = 0
        if self.add_noise_to_chain == True and self.predict_generator_noise == False: stddev = self.noise_stddevs[step]
        if self.add_noise_to_chain == True and self.predict_generator_noise == True: stddev predicted from network
    
        :param input_batch: the output from the previous step (none if this is the first step) 
        :param latent: latent variables, sampled from a unit gaussian
        :param step: the current step in the markov chain
        :param reuse: if we should reuse variables (n.b. we want the same variables for the training and generative 
                versions of the generative network, so sometimes this needs to be true, even in the inhomogeneous case)
        :return: the output sample(s) from the generative network, a stddev for the sample, and the residual connection ratio
        """
        # Compute the encodings of the input, z'_t, if we can
        encodings = None
        if input_batch is not None:
            encodings = self.compute_encodings(input_batch, step, reuse)

        # variable scope setup for decoding/generating
        if self.share_theta_weights and input_batch is not None: # network is different on step 0 (no input_batch)
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

            # If we were given encodings, create a highway/residual connection over the whole step in the chain
            # (encodings[0] = input to the inf network)
            ratio = None
            if encodings is not None:
                ratio = conv2d_t(cur_sample, 1, [4,4], 2, activation_fn=tf.sigmoid)
                ratio = tf.tile(ratio, (1,1,1,self.data_dims[-1]))
                output = tf.multiply(ratio, output) + tf.multiply(1-ratio, encodings[0])

            # Now compute the stddevs for the 
            stddevs = 0
            if self.add_noise_to_chain:
                if self.predict_generator_noise:
                    stddevs = self.stddevs_prediction(output)
                else:
                    stddevs = self.noise_stddevs[step]

            return output, stddevs, ratio





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





    def stddevs_prediction(self, output):
        """
        Used to make a prediction about the stddevs of a mle sample (the output from a generator network). Suppose the 
        sample has a shape of (N, W, H, C), where C is channels and N is the batchsize. We should return a tensor of 
        shape (N,W,H) for the stddevs.

        If the input is of shape (N,W,H,C) then we compute a fully connected layer to W*H*C units

        We reshape to the same shape as the input after, and then expand dims to allow for (tf) broadcasting

        To predict the stddev we stack a few conv layters on the output

        If self.predict_generator_noise_As_scalar, then we're modelling the stddev as a scalar value, rather than 
        a diagonal variance matrix. So add a final fc layer in that case

        :param output: the mle sample that we are going to predict the stddevs for. 
        :return: stddevs, with the same spatial dimensions as output
        """
        stddevs_pred = output
        for i in range(self.predict_generator_stddev_conv_layers):
            stddevs_pred = conv2d_bn_lrelu(stddevs_pred, self.predict_generator_stddev_filter_sizes[i], [4,4], 1)
        stddevs_pred = conv2d(stddevs_pred, num_outputs=1, kernel_size=[1,1], stride=1, activation_fn=tf.sigmoid)
        if self.predict_generator_noise_as_scalar:
            stddevs_shape = stddevs_pred.get_shape().as_list()[1:]
            stddevs_pred_flat = tf.reshape(stddevs_pred, [-1, int(np.prod(stddevs_shape))])
            stddevs_pred_flat = layers.fully_connected(stddevs_pred_flat, 1, activation_fn=tf.sigmoid)
        return stddevs_pred * self.predict_generator_stddev_max





    def generator_flat(self, input_batch, latent, step, reuse=False):
        """
        For the flat test, we use a graphical model of x -> z_0 -> x_0 -> x_1 -> x_2 -> ....
        where x -> z_0 -> x_0 is a VLAE
        and x_0 -> x_1 -> x_2 -> .... are j<ust "flat convolutions" (don't change the shape of the samples x_i).

        Therefore, when we want to "run the flat test", we want on step 0 to use generator_ladder
        and on all other steps, we just want to have a sequence of flat convolutions.

        Thus, we would set self.inference = self.inference_ladder and self.generator = self.generator_flat

        We should also have self.add_noise_to_chain set to true for this test

        Note that we also need to make a new variable for the filter sizes, as the VLAE is still used, and 
        self.filter_sizes is still used by that. So, we make a new array of filter sizes for each "flat step" in the 
        variable self.flat_conv_filter_sizes

        The aim is to understand if there is something inherent about encoding to a (small) latent space that 
        causes the MC network constructed from VLAE's perform poorly.

        The generator network also needs to predict stddevs for the sample, which should work as follows:
        if self.add_noise_to_chain == False: stddev = 0
        if self.add_noise_to_chain == True and self.predict_generator_noise == False: stddev = self.noise_stddevs[step]
        if self.add_noise_to_chain == True and self.predict_generator_noise == True: stddev predicted from network

        As generator flat was a test, we haven't bothered to implsement the self.predict_generator_noise == True case 
    
        :param input_batch: the output from the previous step (none if this is the first step) 
        :param latent: latent variables, sampled from a unit gaussian (only used in step 0 for the VLAE step)
        :param step: the current step in the markov chain
        :param reuse: if we should reuse variables (n.b. we want the same variables for the training and generative 
                versions of the generative network, so sometimes this needs to be true, even in the inhomogeneous case)
        :return: the output sample(s) from the generative network, a stddev for that sample, and the residual connection ratio
        """
        if step == 0:
            return self.generator_ladder(input_batch, latent, step, reuse)

        # variable scope setup 
        if self.share_theta_weights and step != 0: # network is different on step 0 (no input_batch)
            scope_name = "theta/generative_network_flat"
            reuse = tf.AUTO_REUSE
        else:
            scope_name = "theta/generative_flat_step_%d" % step

        # just many flat convolutional layers
        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            hidden_layer = input_batch

            for level in range(self.flat_conv_layers-1):
                hidden_layer = conv2d_bn_lrelu(hidden_layer, self.flat_conv_filter_sizes[level+1], [4,4], 1)

            # Final layer to get the output (normalize to datasets range of values)
            # Return a 0 to be consistent with the 'resnet connection' return val from the generator_ladder function
            output = conv2d(hidden_layer, self.data_dims[-1], [4,4], 1, activation_fn=tf.sigmoid)
            output = (self.dataset.range[1] - self.dataset.range[0]) * output + self.dataset.range[0]
            return output, self.noise_stddevs[step], 0