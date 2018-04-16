import time
from dataset import *

class NoisyTrainer:
    """
    Class encapsulating all of the logic for training a SeqVAE

    Constants used internally defined here (for config):
    pepper_prob = probability of turining a pixel black when adding salt and pepper noise (if added)
    salt_prob = probability of turning a pixel white when adding salt and pepper noise (if added)
    gaussian_noise_scale = scale of gaussian noise added (if added)
    max_training_iter = number of iterations to run, at most, for training
    test_num_iters= number of iters/batches to run through the test loop/visualization loop
    log_loss_freq = the number of training iterations per logging the loss
    """
    pepper_prob = 0.1
    salt_prob = 0.1
    gaussian_noise_scale = 0.1
    max_training_iters = 10000000 # 10000 epochs
    test_num_iters = 5
    log_loss_freq = 20

    def __init__(self, network, dataset, args, logger, base_dir):
        """
        Initialize the trainer for a network.

        Explaination of some variables:
        data_dims = the dimensions of the input. e.g. a RGB images of size 32x32 has data_dims [32,32,3]
        fig = used for visualizations, if/when requested

        Arguments specified in args (specified (with defaults) in main.py):
        denoise_train = if we want to add noise to the training samples, so that the autoencoder learns to denoise also
        vis_frequency = how many training loops per making a visualization
        plot_reconstruction = do we want to plot original/noisy/reconstructed image to models/<network_name>/reconstructon 
                        once per call to test?
        use_gui = if we want to display reconstructions using matplotlib, if running on a desktop

        :param network: the network to be trained by this trainer
        :param dataset: the dataset (in the form of a dataset object) to be used for training (i.e. samples from p_data)
        :param args: a set of ageumtns/parameters to be used in training (defined in main.py)
        :param logger: a logger object, for logging 
        :param base_dir: the base directory to save any data/files to
        :return: none
        """
        self.network = network
        self.dataset = dataset
        self.args = args
        self.batch_size = args.batch_size
        self.data_dims = self.dataset.data_dims
        self.fig = None
        self.LOG = logger
        self.base_dir = base_dir



    def apply_noise(self, original):
        """
        Adds noise to an input if required
        This will unconditionally add noise to a batch of images, and should only be called 
        if we wish to add noise, i.e. if self.args.denoise_train == True


        :param original: the original input/image
        :return: the original, with salt and pepr noise and gaussian noise, if required to add noise
        """
        if not self.args.denoise_train:
            raise Exeption("Called apply_noise, but self.args.denoise_train==False, is this right?")

        # Add salt and pepper noise
        shape = [self.batch_size] + self.data_dims
        noisy_image = np.multiply(original, np.random.binomial(n=1, p=1.0-NoisyTrainer.pepper_prob, size=shape)) + \
                        np.random.binomial(n=1, p=NoisyTrainer.salt_prob, size=shape)

        # Add Gaussian noise
        noisy_image += np.random.normal(scale=NoisyTrainer.gaussian_noise_scale, size=shape)

        # Clip to the allowable range of values
        return np.clip(noisy_image, a_min=self.dataset.range[0], a_max=self.dataset.range[1])



    def train(self):
        """
        The main training loop. 
        Occasionally logs reconstruction loss + calls the test loop
        When we want visualizations, this will call network.visualize to produce them

        :return: nothing
        """
        for iteration in range(NoisyTrainer.max_training_iters):
            iter_beg_time = time.time()

            # occasionally test + visualize
            if iteration % self.args.vis_frequency == 0:
                test_error = self.test(iteration // self.args.vis_frequency)
                self.LOG.info("Reconstruction error per pixel: %d, @ iteration: %d" % (test_error, iteration))
                self.network.visualize(iteration // self.args.vis_frequency)

            # one training iter
            input_batch = self.dataset.next_batch(self.batch_size)
            target_batch = input_batch
            if self.args.denoise_train:
                input_batch = self.apply_noise(input_batch)
            train_loss = self.network.train(input_batch, target_batch)

            # occasionally log losses
            if iteration % NoisyTrainer.log_loss_freq == 0:
                self.LOG.info("Iteration %d: Reconstruction loss %f, time per iter %fs" %
                        (iteration, train_loss, time.time() - iter_beg_time))



    def test(self, epoch, num_iters=test_num_iters):
        """
        Runs through a training loop, and computes the average reconstruction loss
        Also plots the original image, noisy image and reconstructed images to models/<network_name>/reconstruction
        if iter==0 and if specified by self.args.plot_reconstruction

        :param epoch: the training epoch
        :param num_iters: the number of iteratiosn to run the test loop for
        :return: average reconstruction loss from the test loops
        """
        error = 0.0
        for test_iter in range(num_iters):
            test_input_batch = self.dataset.next_test_batch(self.batch_size)
            test_target_batch = test_input_batch
            if self.args.denoise_train:
                test_input_batch = self.apply_noise(test_input_batch)

            reconstruction = self.network.test(test_input_batch)
            error += np.sum(np.square(reconstruction - test_target_batch)) / np.prod(self.data_dims[:2]) / self.batch_size

            # plot original image/noisy image/reconstructed image
            if test_iter == 0 and self.args.plot_reconstruction:
                self.plot_reconstruction(epoch, test_target_batch, test_input_batch, reconstruction)

        return error / num_iters



    def plot_reconstruction(self, train_epoch, original_images, noisy_images, reconstructed_images, num_plot=3):
        """
        Takes a batch of original images, noisy images and reconstructed images, and plots them on a single image 
        so that we can look at it. We only plot 'num_plot' of the images from the batch.

        The constructed image is saved to the directory
        '<working_dir>/models/<network_name>/reconstruction'

        with file name, epochX.png, if X is equal to 'train_epoch'

        :param train_epoch: The epoch we are currently at through training
        :param original_images: A batch of images directly from the dataset
        :param noisy_images: The same as original_images, with *possibly* some noise added to them (if self.args.denoise_train == True)
        :param reconstructed_images: The images output by the autoencoder
        :param num_plot: The number of images to plot from the batch
        :return: None
        """
        border_size = 10
        canvas_width = num_plot * self.data_dims[0]
        canvas_height = 3 * self.data_dims[1] + 2 * border_size
        canvas_depth = self.data_dims[2]
        img_width = self.data_dims[0]
        img_height = self.data_dims[1]


        # black canvas for black and white images, white for RGB
        if original_images.shape[-1] == 1:
            canvas = np.zeros((canvas_width, canvas_height, canvas_depth))
        else:
            canvas = np.ones((canvas_width, canvas_height, canvas_depth))

        # copy all of the images into the canvas, original/noisy/reconstruction
        for img_index in range(num_plot):
            canvas[(img_index * img_width):((img_index+1) * img_width), :img_height] = \
                self.dataset.display(original_images[img_index, :, :])
            canvas[(img_index * img_width):((img_index+1) * img_width), (img_height + border_size):(2*img_height + border_size)] = \
                self.dataset.display(noisy_images[img_index, :, :])
            canvas[(img_index * img_width):((img_index+1) * img_width), (2*img_height + 2*border_size): ] = \
                self.dataset.display(reconstructed_images[img_index, :, :])

        # make the directory if it doesnt exist
        img_folder = self.base_dir + "/reconstruction"
        if not os.path.isdir(img_folder):
            os.makedirs(img_folder)

        # Save the image
        if canvas.shape[-1] == 1:
            misc.imsave(os.path.join(img_folder, 'current.png'), canvas[:, :, 0])
            misc.imsave(os.path.join(img_folder, 'epoch%d.png' % train_epoch), canvas[:, :, 0])
        else:
            misc.imsave(os.path.join(img_folder, 'current.png'), canvas)
            misc.imsave(os.path.join(img_folder, 'epoch%d.png' % train_epoch), canvas)

        # If we want a GUI with the images, update the gui now
        if self.args.use_gui:
            if self.fig is None:
                self.fig, self.ax = plt.subplots()
                self.fig.suptitle("Reconstruction of " + str(self.network.name))
            self.ax.cla()
            if canvas.shape[-1] == 1:
                self.ax.imshow(canvas[:, :, 0], cmap=plt.get_cmap('Greys'))
            else:
                self.ax.imshow(canvas)
            plt.draw()
            plt.pause(1)

