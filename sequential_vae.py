from abstract_network import *
from scipy import misc


class SequentialVAE(Network):
    """
    Implementation of SequentialVAE, extending our abstract Network class.

    Description:
    TODO: Add a full description of what this code will/does do
    TODO: define p_data, q(z|x), p(x|z), x_i, z_i and so on
    TODO: describe the relations between each of them
    TODO: When we can add different descriptions for different modules
    TODO: Reference variational ladder auto encoder + define notation there + describe its usage in our SeqVAE
    TODO: explain the the output of encoder is a mean and variance (so a probability distribution)
    TODO: explain that the output of the decoder is an x, and that implicitly defines 


    Work to be done:
    ---THU---
    TODO: debug this code and double check that it actually works...
    TODO: run tensorboard + check resnet weights being used
    TODO: Try with very low capacity networks + try with much higher capacity networks
    +++++
    TODO: Add a debug flag
    TODO: Add more tensorboard for debugging (e.g. ratio of update magnitude to weight magnitude + weight of regularization term vs reconstruction + all things had in 273b/suggested by 231n)
    <Just copy over>
    +++++
    Somehow check that the 
    DUBUG PRINT all of the variables

    ---SAT---
    TODO: Add variable length MCs for training and for testing too
    TODO: Update self.steps (specify that we should set it to None to specify this)

    ---MON---
    TODO: (Should cover the below 5 todos) Update the networks to add the extra dependencies and try out all of the combinations in our draft math
    TODO: (Should cover the below 5 todos) Update the loss functions according to the draft math
    TODO: ADD tensorboard variables to look at each of these additional loss functions + be able to tell how much of a difference it is making

    TODO: Add an extra objective to maximize the jensens inequality error term when predicting the loss of the reconstruction
    TODO: -> this is basically adding the dependence between z's? Look at original code when doing this "use_latent_pred", "condition" and "latent_pred" stuff
    TODO: Check math is consistent with vlae math (justify the differences)
    TODO: Explicitly try decorrelating the z's as suggested to me
    TODO: this section is essentially making the latent code depend on x or (x and z_i's) or (x and z_i's and x_j's) etc
    TODO: Compare what I want to do (optimize to maximize error term in Jenson's) 


    TODO: Additional debugging stuff -> allow different sizes of convolution in the ladders, print warnings if something goes weird
            (Compute the next image size and stride and filter size etc from what was specified)

    ---Further---
    TODO: With the additional objective function, that should just update phi and not theta?

    TODO: Try the different combinations of parameter sharing, try to work out the effect on each

    TODO: Try to investigate the different ways that the latent space could be modelled
    TODO: Try a prior of uniform (or a wide gaussian) on z_i, and then hopefully q_psi(z_i|z_<i,x_i-1) can be a very specific gaussian with different mu and sigma's
    TODO: try to experiment more with the latent space being a mixture of gaussians (try to make the prob distr very spiky?)

    TODO: as we are using a chain, maybe we can output a varience from the decoder, and that can provide some useful information

    TODO: Analysis framework -> producing graphs etc SeaBorne etc (graph of reconstruction loss + values of the "error terms etc")
    """
    def __init__(self, dataset, batch_size, name, logger):
        """
        Initialization of the network, defines parameters and constructs the network in tenforflow
        SEE ALSO: __init__ from abstract_network.py, which defines our superclass 'Network'

        Description of parameters defined:
        (VLAE params)
        self.vlae_levels = the number of levels in each VLAE used in the seqvae
                    technically, "one level" here is two convolutional layers, which reduce the image size by half (step sizes of 2 then 1)
                    (the last level has a fully connected layer too)
        self.vlae_latent_dims = a list of the number of dimensions in the latent space for each part of the ladder in the vlae
        self.image_sizes = the size of the "intermediate images". A list of 'self.vlae_levels + 1' integers.
        self.filter_sizes = the size of the filters. A list of length 'self.vlae_levels + 2', as there are 'self.VLAE_levels + 1' 
                    intermediate representations (including input) and we have two for the output level (includes a fully connected layer)

        (SeqVAE params)
        self.share_recognition_params = should each recognition network, q_phi(z_i|x_i), share parameters phi?
        self.share_generative_params = should each generative network, p_theta(x_i+1|z_i), share parameters theta?
        self.share_encoder_params = should the encoder networks (x->z_t and x_t-1,_ ->x_t) share parameters?
        self.share_latent_code_params = should each latent code network, q_psi(z_i|x) share parameters psi?
        self.inference = the recognition network to use (currently just VLAE's recognition network is an option)
                    N.B. self.inference is a functional value
        self.generator = the generator network to use (currently just VLAE's generator is an option)
                    N.B. self.generator is a functional value
        self.mc_steps = the number of steps to use in the markov chain
        self.latent_dim = the dimension of the latent spaces (i.e. the number of dimensions z_i has for EACH i)
        self.intermediate_reconstruction = if each x_i should be a reconstruction of x. (If we should add a reconstruction loss for each x_i)
        self.combine_noise_method = 'concat'/'add'/'gated_add', and specifies how to add in (latent) noise into the embeddings of the autoencoder 
                    (see combined_noise for more detail)

        'self.combine_noise_method' should take one of the following values:
        "concat": Concatenate the noise onto the embeddings
        "add": directly add the noise and the embeddings
        "gated_add": multiply the noise by a (trainable) variable, and then add

        (Hyperparams)
        self.learning_rate = the learning rate to use during training
        self.reg_coeff_rate = decides the rate of which we increase the (coefficient of) regularization on latent variables from 0 to 1 over time

        (Training params)
        self.save_freq = how frequently to save the network during training
        self.tb_summary_freq = how frequently to write summaries to tensorboard

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
        self.filter_sizes = [self.data_dims[-1], 48, 96, 192, 384, 512]

        # SeqVAE parameters 
        self.share_recognition_params = False
        self.share_generative_params = False
        self.share_encoder_params = False
        self.share_latent_code_params = False
        self.inference = self.inference_ladder 
        self.generator = self.generator_ladder 
        self.mc_steps = 8
        self.latent_dim = np.sum(self.vlae_latent_dims)
        self.intermediate_reconstruction = True
        self.combine_noise_method = "concat"

        # Hyperparams and training params
        self.learning_rate = 0.0002
        self.reg_coeff_rate = 5000.0 # 1 epoch = 1000
        self.save_freq = 2000
        self.tb_summary_freq = 10

        # Config for different netnames, where customization is needed.
        # add overides for any of the above parameters here
        if self.name == "sequential_vae_celebA_inhomog":
            # nothing
            pass

        elif self.name == "sequential_vae_celebA_homog":
            self.share_generative_params = True

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
            self.steps = 5
            self.generator = self.generator_ladder
            self.inference = self.inference_ladder

        elif self.name == "sequential_vae_mnist_homog":
            self.vlae_levels = 3
            self.vlae_latent_dims = [8, 8, 8]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]
            self.steps = 5
            self.generator = self.generator_ladder
            self.inference = self.inference_ladder
            self.share_generative_params = True

        elif self.name == "sequential_vae_mnist_share_all":
            self.vlae_levels = 3
            self.vlae_latent_dims = [8, 8, 8]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]
            self.steps = 5
            self.generator = self.generator_ladder
            self.inference = self.inference_ladder
            self.share_generative_params = True
            self.share_recognition_params = True

        elif self.name == "sequential_vae_mnist_share_encoders":
            self.vlae_levels = 3
            self.vlae_latent_dims = [8, 8, 8]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]
            self.steps = 5
            self.generator = self.generator_ladder
            self.inference = self.inference_ladder
            self.share_encoder_params = True

        elif self.name == "sequential_vae_mnist_share_inference":
            self.vlae_levels = 3
            self.vlae_latent_dims = [8, 8, 8]
            self.latent_dim = np.sum(self.vlae_latent_dims)
            self.image_sizes = [32, 16, 8, 4] 
            self.filter_sizes = [self.data_dims[-1], 64, 128, 192, 256]
            self.steps = 5
            self.generator = self.generator_ladder
            self.inference = self.inference_ladder
            self.share_recognition_params = True

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

        Tensorflow variables defined here:
        self.input_placeholder = placeholder for the input image (= the target image + optional noise)
        self.target_placeholder =  placeholder for the target image (= input image, without any noise)
        self.reg_coeff = placeholder (with default value) for the coefficient of regularization
        
        self.latents = placeholders for latent variables, use these when we want to use the network generatively
        self.training_samples = samples generated along the chain, use these one's when training the network
                    this uses a latent variable that's sampled using the mean and stddev from inference network of the previous sample
        self.generator_samples = samples generated along the chaing, use these one's when running the generatively
                    this uses the latent variable fed into the placeholder in self.latents

        N.B. Although when making the generator and training sample variables, we make two seperate 
        calls to 'self.generator', parameter sharing (through variable scopes) 
    
        self.loss = the total loss, the value that sums all of the lasses from each stage in the network and which we optimize over
        self.final_loss = the loss of the final sample (i.e. the output of the markov chain). i.e. how good our performance is

        self.merged_summaries = a tensorflow op, which combines all previous tensorboard summaries, so we can run all of them at the same time (in a tf session)
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
            # Create placeholder for latent variable on this step. i.e. z_i
            latent_placeholder = tf.placeholder(shape=[None, self.latent_dim], dtype=tf.float32, name=("latent_placeholder_%d" % step))
            self.latents.append(latent_placeholder)

            # On the first step, let x_0 be uniform random noise
            if step == 0:
                generator_sample = tf.random_uniform(shape=tf.stack([tf.shape(self.input_placeholder)[0]] + self.data_dims))
                self.generator_samples.append(generator_sample)

            # Run the recognition network (encoder) to get mean and stddev of z_i. then sample a z_i
            latent_mean, latent_stddev = self.inference(self.input_placeholder, step)
            latent_sample = latent_mean + tf.multiply(latent_stddev, 
                                                tf.random_normal(tf.stack([tf.shape(self.input_placeholder)[0], self.latent_dim])))

            # Generate the next sample. Make two different variables (using the same networks with different inputs) for training/generation
            # And keep track of (tensorboard summaries) the residual connection weights (for debugging)
            # In inhomogenous case, generator_samples need to share params from training. (just this time step)
            # If self.share_generative_params == True, then we share all of them automatically anyway (across all time steps)
            if step == 0:
                training_sample = self.generator(None, latent_sample, step)
                generator_sample = self.generator(None, latent_placeholder, step, reuse=True)
            else:
                training_sample, resnet_ratios = self.generator(training_sample, latent_sample, step)
                generator_sample, _ = self.generator(generator_sample, latent_placeholder, step, reuse=True)
                tf.summary.scalar("resnet_gate_weight_step_%d" % step, tf.reduce_mean(resnet_ratios))
            self.training_samples.append(training_sample)
            self.generator_samples.append(generator_sample)

            # Construct the loss for this step. (KL distance for regularizing the latent code and squared reconstruction loss for sample)
            reconstruction_loss = tf.reduce_mean(tf.square(training_sample - self.target_placeholder))
            regularization_loss = -0.5 * self.latent_dim + tf.reduce_mean(-tf.log(latent_stddev) +
                                                0.5 * tf.square(latent_stddev) +
                                                0.5 * tf.square(latent_mean)) 

            # Add to the overall loss
            if self.intermediate_reconstruction or step == self.mc_steps-1:
                self.loss += 16 * reconstruction_loss 
            self.loss += self.reg_coeff * regularization_loss

            # Keep track of the final reconstruction error (we just set this each time)
            self.final_loss = reconstruction_loss

            # Add tensorboards summaries for the losses at this step
            tf.summary.scalar("reconstruction_loss_step_%d" % step, reconstruction_loss)
            tf.summary.scalar("regularization_loss_step_%d" % step, regularization_loss)

            # Finally, prevent gradients from propogating between the different samples, if each encoder/decoder have their own losses
            if self.intermediate_reconstruction:
                training_sample = tf.stop_gradient(training_sample)

        # Add tensorboard summary for the loss
        tf.summary.scalar("loss", self.loss)

        # Group all summaries into one variable we can keep hold of 
        self.merged_summary = tf.summary.merge_all()

        # Finally, make the train op (the optimizer)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)





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





    def train(self, batch_input, batch_target):
        """
        Perofrm ONE training update. (This is called from the main training loop in trainer.py)
        Also this will 

        :param batch_input: The input to the network for training
        :param batch_target: The target for the network (batch_target = batch_input - (noise if any))
        :return: The reconstruction loss of the FINAL sample (i.e. how well did the network do this time?)
        """
        self.iteration += 1

        # run values through our session 
        feed_dict = {self.input_placeholder: batch_input,
                     self.reg_coeff: 1 - math.exp(-self.iteration / self.reg_coeff_rate),
                     self.target_placeholder: batch_target}
        _, _, final_reconstruction_loss = self.sess.run([self.train_op, self.loss, self.final_loss], feed_dict=feed_dict)

        # Occasionally save the network params + write summaries to tensorboard
        if self.iteration % self.save_freq == 0:
            self.save_network()
        if self.iteration % self.tb_summary_freq == 0:
            self.LOG.debug("Writing tensorboard summaries, iter %d" % self.iteration)
            summary = self.sess.run(self.merged_summary, feed_dict=feed_dict)
            self.writer.add_summary(summary, self.iteration)

        # return the final reconstruction loss (averaged per pixel)
        return final_reconstruction_loss / self.data_dims[0] / self.data_dims[1]





    def test(self, batch_input):
        """
        Runs the training network, 

        :param batch_input: The input to the network, noise that we want to turn into nice images
        :return: The final output of the SeqVAE network, i.e. the generated image
        """
        feed_dict = {self.input_placeholder: batch_input}
        generated_image = self.sess.run(self.training_samples[-1], feed_dict=feed_dict)
        return generated_image





    def generate_mc_samples(self, batch_input, batch_size=None):
        """
        Run the network in a generative mede. Generate sample from a normal distribution and feed 
        them into the network (as the latent variables). To run the generative part(s) of the network
        we run the self.generator_samples tf ops.

        N.B. This isn't used for optimization, just for visualization.

        :param batch_input: the input to the network (random noise) (None => use self.batch_size)
        :param batch_size: the size of the back
        :return: The generated samples from running the network in generative mode
        """
        if batch_size is None:
            batch_size = self.batch_size

        feed_dict = dict()
        feed_dict[self.input_placeholder] = batch_input
        for i in range(self.mc_steps):
            feed_dict[self.latents[i]] = np.random.normal(size=(batch_size, self.latent_dim))

        output = self.sess.run(self.generator_samples, feed_dict=feed_dict)
        return output





    def training_mc_samples(self, batch_input):
        """ 
        Run the network to generate samples in the training mode. To run the training part(s) 
        of the network we run the self.training_samples tf ops.

        N.B. This isn't used for optimization, just for visualization.

        :param batch_input: the input to the network (samples from the training set)
        :return: The generated samples from running the network in training mode
        """
        feed_dict = dict()
        feed_dict[self.input_placeholder] = batch_input
        output = self.sess.run(self.training_samples, feed_dict=feed_dict)
        return output





    def visualize(self, epoch, batch_size=10, use_gui=True):
        """
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
                    v[b*self.data_dims[0]:(b+1)*self.data_dims[0], t*self.data_dims[1]:(t+1)*self.data_dims[1]] = self.dataset.display(z[t][b])

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



    def inference_ladder(self, inputs, step, reuse=False):
        """
        x -> z_t

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

        :param inputs: the input to encode (in the math this is x)
        :param step: the step in the overall markov chain (used for variable scoping)
        :param reuse: if we should reuse variables (n.b. we want the same variables for the training and generative 
                versions of the generative network, so sometimes this needs to be true, even in the inhomogeneous case)
        :return: the mean(s) and std_dev(s) of the latent state
        """
        if self.share_recognition_params:
            scope_name = "inference_network"
            reuse = tf.AUTO_REUSE
        else:
            scope_name = "inference_step_%d" % step

        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            cur_encoding = inputs
            image_sizes = [cur_encoding.get_shape().as_list()[1]]
            latent_mean = []
            latent_stddev = []

            for level in range(self.vlae_levels-1):
                # encoding steps (move to next level in ladder)
                hidden = conv2d_bn_lrelu(cur_encoding, self.filter_sizes[level+1], [4,4], 2)
                cur_encoding = conv2d_bn_lrelu(hidden, self.filter_sizes[level+1], [4,4], 1)

                # latent code at this level in the ladder
                ladder = tf.reshape(cur_encoding, [-1, np.prod(cur_encoding.get_shape().as_list()[1:])])
                ladder_mean = tf.contrib.layers.fully_connected(ladder, self.vlae_latent_dims[level], activation_fn=tf.identity)
                ladder_stddev = tf.contrib.layers.fully_connected(ladder, self.vlae_latent_dims[level], activation_fn=tf.sigmoid)

                # maintain lists variables
                image_sizes.append(cur_encoding.get_shape().as_list()[1])
                latent_mean.append(ladder_mean)
                latent_stddev.append(ladder_stddev)

            # Add the last level (only have one convolution and a fully connected layer)
            cur_encoding = conv2d_bn_lrelu(cur_encoding, self.filter_sizes[self.vlae_levels-1], [4,4], 2)
            image_size = cur_encoding.get_shape().as_list()[1]
            cur_encoding = tf.reshape(cur_encoding, [-1, np.prod(cur_encoding.get_shape().as_list()[1:])])
            cur_encoding = fc_bn_lrelu(cur_encoding, self.filter_sizes[self.vlae_levels])

            ladder_mean = tf.contrib.layers.fully_connected(ladder, self.vlae_latent_dims[self.vlae_levels-1], activation_fn=tf.identity)
            ladder_stddev = tf.contrib.layers.fully_connected(ladder, self.vlae_latent_dims[self.vlae_levels-1], activation_fn=tf.sigmoid)

            image_sizes.append(image_size)
            latent_mean.append(ladder_mean)
            latent_stddev.append(ladder_stddev)

            # Check that 'self.image_sizes' is actually correct. Log an error and quit if it's not (generative network will not work)
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





    def generator_ladder(self, inputs, latent, step, reuse=False):
        """
        x_t-1, z_t -> x_t

        Generative network part of the VLAE. Basically the same architecture as the inference network
        but run in reverse (i.e. using transposed convolutions rather than convolutions).

        Inference encodings are computed from the input (x_t-1, the output from the previous step) 
        and we add shortcuts between the ith level of the inference network and the ith level of the 
        generative network 

        We add residual connections over the whole autoencoder

        The input to the lowest level (i.e. level == self.vlae_levels) is the latent state

        The input to all other levels is the output of the previous level, with the latent 
        state being added using 'self.combine_noise'

        For all levels, if inputs is not null, we add shortcuts. Meaning that we directly 
        add the encoding at the ith layer of the reconition netowork to the ith level 
        of the generator network
    
        :param inputs: the output from the previous step (none if this is the first step) 
        :param latent: latent variables, sampled from a unit gaussian
        :param step: the current step in the markov chain
        :param reuse: if we should reuse variables (n.b. we want the same variables for the training and generative 
                versions of the generative network, so sometimes this needs to be true, even in the inhomogeneous case)
        :return: the output sample(s) from the generative network, and the residual connection ratio
        """
        encodings = None
        if inputs is not None:
            encodings = self.compute_encodings(inputs, step, reuse)

        # variable scope setup for decoding/generating
        if self.share_generative_params and step != 0: # network is different on step 0 (no inputs)
            scope_name = "generative_network"
            reuse = tf.AUTO_REUSE
        else:
            scope_name = "generative_step_%d" % step

        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            # split up the latent variables
            ladder = tf.split(latent, self.vlae_latent_dims, 1)
            for i in range(self.vlae_levels):
                ladder[i] = tf.reshape(ladder[i], [-1, self.vlae_latent_dims[i]])

            # project them into the correct image sizes to be added/concatenated to the intermediate representations
            for i in range(self.vlae_levels-1):
                ladder[i] = fc_bn_lrelu(ladder[i], self.image_sizes[i+1] * self.image_sizes[i+1] * self.filter_sizes[i+1])
                ladder[i] = tf.reshape(ladder[i], [-1, self.image_sizes[i+1], self.image_sizes[i+1], self.filter_sizes[i+1]])
            ladder[self.vlae_levels-1] = fc_bn_lrelu(ladder[self.vlae_levels-1], self.filter_sizes[self.vlae_levels+1])
            ladder[self.vlae_levels-1] = tf.reshape(ladder[self.vlae_levels-1], [-1, self.filter_sizes[self.vlae_levels+1]])

            # First level of generator network / last level of ladder
            if encodings is not None:
                cur_sample = encodings[self.vlae_levels]
                cur_sample = self.combine_noise(cur_sample, ladder[self.vlae_levels-1])
            else:                
                cur_sample = ladder[self.vlae_levels-1]

            shape = [self.image_sizes[self.vlae_levels], self.image_sizes[self.vlae_levels], self.filter_sizes[self.vlae_levels]]
            cur_sample = fc_bn_lrelu(cur_sample, np.prod(shape))
            cur_sample = tf.reshape(cur_sample, [-1] + shape)
            
            # Middle layers of the network (each iteration deals with 1 intermediate image size)
            # 2 transposed convolutions, adding in a shortcut from the inference encoder to the first if we have it
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

            # Return output, adding resnet connection (encodings[0] = input to inf network)
            if encodings is not None:
                ratio = conv2d_t(cur_sample, 1, [4,4], 2, activation_fn=tf.sigmoid)
                ratio = tf.tile(ratio, (1,1,1,self.data_dims[-1]))
                output = tf.multiply(ratio, output) + tf.multiply(1-ratio, encodings[0])
                return output, ratio
            else:
                return output



    def compute_encodings(self, inputs, step, reuse=False):
        """
        TODO: Description

        :param inputs: the output from the previous step (none if this is the first step) 
        :param step: the current step in the markov chain
        :param reuse: if we should reuse variables (n.b. we want the same variables for the training and generative 
                versions of the generative network, so sometimes this needs to be true, even in the inhomogeneous case)
        :return: Encodings for this given input
        """
        # variable scope setup for encoding input
        if self.share_encoder_params and self.share_recognition_params:
            scope_name = "inference_network"
            reuse = True
        elif self.share_encoder_params:
            scope_name = "inference_step_%d" % step
            reuse = True
        if self.share_generative_params: 
            scope_name = "generative_encoder_network"
            reuse = tf.AUTO_REUSE
        else:
            scope_name = "generative_encoder_step_%d" % step

        # compute encodings from the inputs (follows same structure as inference_ladder)
        with tf.variable_scope(scope_name, reuse=reuse) as scope:
            cur_encoding = inputs
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

