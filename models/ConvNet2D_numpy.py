import numpy as np

class ConvNet2D:
    def __init__(self, in_channels, out_channels, num_filters, filter_sizes, strides, paddings, 
                 pooling=None, pool_size=None):
        """CNN classifier with parameterizable number of convolutional layers and pooling layers, and a fully connected layer"""

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.strides = strides
        self.paddings = paddings
        self.pooling = pooling
        self.pool_size = pool_size

        assert len(num_filters) == len(filter_sizes) == len(strides) == len(paddings)
        assert pooling is None or len(pooling) == len(pool_size)

        self.conv_layers = []
        self.pooling_layers = []

        for i in range(len(num_filters)):
            self.conv_layers.append(Conv2d(in_channels, num_filters[i], filter_sizes[i], strides[i], paddings[i]))
            in_channels = num_filters[i]  # Update in_channels for next layer

            if pooling is not None and pooling[i]:
                self.pooling_layers.append(Pooling(pool_size[i]))

        self.fc_weights = np.random.randn(X.shape[1], self.out_channels) * np.sqrt(2.0 / X.shape[1])
        self.fc_biases = np.zeros((1, self.out_channels))

    def update_parameters(self, grads, learning_rate):
        """Update the weights using the gradients"""
        for i in range(len(self.conv_layers)):
            self.conv_layers[i].W -= learning_rate * grads[f'dW{i}']
            self.conv_layers[i].b -= learning_rate * grads[f'db{i}']

        self.fc_weights -= learning_rate * grads['dW_fc']
        self.fc_biases -= learning_rate * grads['db_fc']

    @staticmethod
    def sigmoid(self, X):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-X))

    def forward(self, X):
        """Forward pass through the CNN"""
        for i in range(len(self.conv_layers)):
            X = self.conv_layers[i].forward(X)
            if len(self.pooling_layers) > i:
                X = self.pooling_layers[i].forward(X)

        # Flatten the output for the fully connected layer
        X = X.reshape(X.shape[0], -1)

        # Fully connected forward pass
        X = np.dot(X, self.fc_weights) + self.fc_biases
        X = self.sigmoid(X)
        return X


    def fit(self, X, Y, learning_rate=0.01, n_iters=1000, seed=0):
        '''Fit according to the learning rate and number of iterations'''
        np.random.seed(0)
        m = X.shape[1]
        input_dim = X.shape[0]
        output_dim = 1 
        costs = []

        for i in range(n_iters):
            A2, cache = self.forward(X)
            cost = self.compute_cost(A2, Y)
            grads = self.backward(cache, X, Y)
            self.parameters = self.update_parameters(grads, learning_rate)
            costs.append(cost)
            # Print the cost every 100 iterations
            if i % 100 == 0:
                print(f'Cost after iteration {i}: {cost}')

        return self.parameters, costs
    


    def compute_cost(self, AL, Y):
        pass


    def fit(self, X, Y, optimizer, learning_rate, n_iters, seed=0):
        '''Fit according to the learning rate and number of iterations'''
        np.random.seed(seed)
        costs = []

        self.initialize_parameters()

        v = {}
        s = {}
        t = 0
        for l in range(1, len(self.layers_dims)):
            v["dW" + str(l)] = np.zeros_like(self.parameters["W" + str(l)])
            v["db" + str(l)] = np.zeros_like(self.parameters["b" + str(l)])
            s["dW" + str(l)] = np.zeros_like(self.parameters["W" + str(l)])
            s["db" + str(l)] = np.zeros_like(self.parameters["b" + str(l)])

        for i in range(n_iters):

            AL, caches = self.forward_propagation(X)
            cost = self.compute_cost(AL, Y)
            grads = self.backward_propagation(AL, Y, caches)
            if optimizer == "adam":
                t += 1
                self.update_parameters_adam(grads, learning_rate, v, s, t)
            costs.append(cost)

            # Print the cost every 100 iterations
            if i % 100 == 0:
                print(f'Cost after iteration {i}: {cost}')

        return self.parameters, costs


class Pooling:

    def __init__(self, f, stride, mode = 'max'):

        self.hparameters = {"f":f ,"stride": stride}
        self.mode = mode

        assert self.mode in ['max', 'average']
    
    def forward(self, input):
        """
        Implements the forward pass of the pooling layer
        
        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        hparameters -- python dictionary containing "f" and "stride"
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
        
        Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
        """
        
        # Retrieve dimensions from the input shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        # Retrieve hyperparameters from "hparameters"
        f = self.hparameters["f"]
        stride = self.hparameters["stride"]
        
        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev
        
        # Initialize output matrix A
        A = np.zeros((m, n_H, n_W, n_C))              
        
        for i in range(m):                         # loop over the training examples
            for h in range(n_H):                     # loop on the vertical axis of the output volume
                #Find the vertical start and end of the current "slice"
                vert_start = h*stride
                vert_end = h*stride + f
                
                for w in range(n_W):                 # loop on the horizontal axis of the output volume
                    #Find the vertical start and end of the current "slice"
                    horiz_start = w*stride
                    horiz_end = w*stride + f
                    
                    for c in range (n_C):            # loop over the channels of the output volume
                        
                        #Use the corners to define the current slice on the ith training example of A_prev, channel c
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]  
                        
                        #Compute the pooling operation on the slice. 
                        if mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)
        
        
        # Store the input and hparameters in "cache" for pool_backward()
        cache = (A_prev, hparameters)
        
        # Making sure your output shape is correct
        assert(A.shape == (m, n_H, n_W, n_C))
        
        return A, cache

    def backward(dA, cache):
        """
        Implements the backward pass of the pooling layer
        
        Arguments:
        dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
        cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
        
        Returns:
        dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
        """
        # Retrieve information from cache
        (A_prev, hparameters) = cache
        
        f = hparameters["f"]
        stride = hparameters["stride"]
        
        # Retrieve dimensions from A_prev's and dA's
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        m, n_H, n_W, n_C = dA.shape
        
        # Initialize dA_prev
        dA_prev = np.zeros(A_prev.shape)
        
        for i in range(m): # loop over the training examples
            
            a_prev_i = A_prev[i] #training example from A_prev 
            
            for h in range(n_H):                   # loop on the vertical axis
                for w in range(n_W):               # loop on the horizontal axis
                    for c in range(n_C):           # loop over the channels (depth)
            
                        #Find the corners of the current "slice"
                        vert_start = h*stride
                        vert_end = h*stride + f
                        horiz_start = w*stride
                        horiz_end = w*stride + f
                        
                        #Compute the backward propagation in both modes.
                        if self.mode == "max":
                            
                            #Use the corners and "c" to define the current slice from a_prev
                            a_prev_slice =  a_prev_i[vert_start:vert_end, horiz_start:horiz_end, c] 

                            # Only the maximum value in each pooling window influences the output.
                            # We use a mask to determine which element in the input slice was selected as max.
                            # We assign the gradient only to that max value in dA_prev.
                                                        
                            mask = (a_prev_slice == np.max(a_prev_slice))

                            #Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += dA[i, h, w, c] * mask
                            
                        elif self.mode == "average":
                            
                            #Get the value da from dA (≈1 line)
                            da = dA[i, h, w, c]

                            #Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da.

                            #Each element in the input pooling window equally influences the output.
                            #The gradient is evenly distributed across all elements in the pooling region.
                            average = da / (f * f)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += np.full((f, f), average)

        
        # Making sure your output shape is correct
        assert(dA_prev.shape == A_prev.shape)
        
        return dA_prev


class Conv2d:

    def __init__(in_channels, out_channels, kernel_size, stride=1, padding_mode='zeros'):
        """
        Applies a 2D convolution over an input signal composed of several input planes.

        Arguments:
        in_channels -- Number of channels in the input image
        out_channels -- Number of channels produced by the convolution
        kernel_size -- Size of the convolving kernel
        stride -- Stride of the convolution. Default: 1
        padding_mode -- Padding mode. Default: 'zeros'
        
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_filters
        self.hparameters = {"stride": stride, "pad": padding}
        self.W = np.random.randn(self.kernel_size, self.n_filters) * 0.01 
        self.b = np.random.randn(self.n_filters, 1)


    def zero_pad(self, X):
        """
        Pad with zeros all 2D data of the dataset X. The padding is applied to both dimensions

        Arguments:
        X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m 2D objects
        pad -- integer, amount of padding around each object on both dimensions

        Returns:
        X_pad -- padded object of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
        """
        return np.pad(X, ((0, 0), (self.pad, self.pad), (self.pad, self.pad), (0, 0)), 'constant', constant_values=(0, 0))

    def conv_single_step(self, slice):
        """
        Apply one filter defined by parameters W on a single slice (slice) of the output activation of the previous layer.

        Arguments:
        slice -- slice of input data of shape (f, f, n_C_prev)
        W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
        b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

        Returns:
        Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
        """
        return np.sum(np.multiply(slice, self.W) + self.b)


    def forward(A_prev):
        """
        Implements the forward propagation for a convolution function
        
        Arguments:
        A_prev -- output activations of the previous layer, 
            numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
        b -- Biases, numpy array of shape (1, 1, 1, n_C)
        hparameters -- python dictionary containing "stride" and "pad"
            
        Returns:
        Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward() function
        """
        
        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve dimensions from W's shape (≈1 line)
        (f, f, n_C_prev, n_C) = W.shape


        stride = self.hparameters["stride"]
        pad = self.hparameters["pad"]

        # Compute the dimensions of the CONV output volume using the formula. 
        n_H = int(np.floor((n_H_prev - f + 2 * pad) / stride) + 1)
        n_W = int(np.floor((n_W_prev - f + 2 * pad) / stride) + 1)

        # Initialize the output volume Z with zeros
        Z = np.zeros((m, n_H, n_W, n_C))

        # Create A_prev_pad by padding A_prev
        A_prev_pad = zero_pad(A_prev, pad)

        for i in range(m):               # loop training examples
            for h in range(n_H):           # loop over vertical axis of the output volume
                #Find the vertical start and end of the current "slice"
                vert_start = h * stride
                vert_end = vert_start + f

                for w in range(n_W):       # loop over horizontal axis of the output volume
                    #Find the horizontal start and end of the current "slice"
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    for c in range(n_C):   # loop over channels (= #filters) of the output volume

                        #Use the corners of the training example i to define the (3D) slice of a_prev_pad. 
                        a_slice_prev = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]

                        #Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                        weights = W[:,:,:,c]       #(f, f, n_C)
                        biases = b[:,:,:,c]        #(1, 1, 1)
                        Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
                        # Activation
                        A[i, h, w, c] = activation_func(Z[i, h, w, c])


        # Save information in "cache" for the backprop
        Z_cache = (A_prev, W, b, hparameters)
        A_cache = Z

        return A, (linear_cache, activation_cache)

    def backward(dZ, cache):
        """
        Implement the backward propagation for a convolution function
        
        Arguments:
        dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward(), output of conv_forward()
        
        Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        dW -- gradient of the cost with respect to the weights of the conv layer (W)
            numpy array of shape (f, f, n_C_prev, n_C)
        db -- gradient of the cost with respect to the biases of the conv layer (b)
            numpy array of shape (1, 1, 1, n_C)
        """    
        
        (A_prev, W, b, hparameters) = cache         # Retrieve information from "cache"
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape         # Retrieve dimensions from A_prev's shape
        (f, f, n_C_prev, n_C) = W.shape         # Retrieve dimensions from W's shape
        
        stride = hparameters["stride"]
        pad = hparameters["pad"]
        
        (m, n_H, n_W, n_C) = dZ.shape  # Retrieve dimensions from dZ's shape

        # Initialize dA_prev, dW, db 
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                         
        dW = np.zeros((f, f, n_C_prev, n_C)) 
        db = np.zeros((1, 1, 1, n_C)) 
        
        # Pad A_prev and dA_prev
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
        
        for i in range(m):  # loop over the training examples
            
            #select ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad_i = A_prev_pad[i]
            da_prev_pad_i = dA_prev_pad[i]
            
            for h in range(n_H):                   # loop over vertical axis of the output volume
               for w in range(n_W):               # loop over horizontal axis of the output volume
                   for c in range(n_C):           # loop over the channels of the output volume
                        
                        #Find the corners of the current "slice"
                        vert_start = h * stride
                        vert_end = h * stride + f
                        horiz_start = w * stride
                        horiz_end = w * stride + f

                        #Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad_i[vert_start:vert_end, horiz_start:horiz_end, :]

                        #Update gradients for the window and the filter's parameters

                        #formula for computing  𝑑𝐴 with respect to the cost for a certain filter 𝑊𝑐 and a given training example
                        da_prev_pad_i[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c] 
                        dW[:,:,:,c] += a_slice * dZ[i, h, w, c]  #formula for computing dw (derivative of one filter 𝑊𝑐 with respect to loss)
                        db[:,:,:,c] += dZ[i, h, w, c] #formula for computing db (derivative of one filter 𝑊𝑐 with respect to loss)
                        
            #Set the ith training example's dA_prev to the unpadded da_prev_pad
            dA_prev[i, :, :, :] = da_prev_pad_i[pad:-pad, pad:-pad, :]

        # Making sure output shape is correct
        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
        
        return dA_prev, dW, db


    

class FullyConnected:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim #(n_x)
        self.output_dim = output_dim #(n_y)
        self.W = np.random.randn(self.output_dim, self.input_dim) * 0.01 # shape: (n_y, n_x)
        self.b = np.zeros((self.output_dim, 1)) # shape: (n_y, 1)


    def forward_linear(self, X):
        '''Forward pass through the
        linear layer'''
        Z = np.dot(self.W, X) + self.b

        W , b = self.W.copy(), self.b.copy()
        linear_cache = (X, W, b)
        return Z, linear_cache

    def forward_activation(self, Z):
        '''Forward pass through the
        activation function'''
        A = self.softmax(Z)
        activation_cache = Z
        return A, activation_cache

    @staticmethod
    def softmax(X):
        """Stable Softmax Activation"""
        exps = np.exp(X - np.max(X, axis=0, keepdims=True))  # Prevent numerical instability
        return exps / np.sum(exps, axis=0, keepdims=True)


    def forward(self, X):
        """
        Implement forward propagation for the LINEAR->SOFTMAX computation
        
        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        
        Returns:
        AL -- activation value from the output (last) layer
        caches -- list of caches containing:
                    linear_cache -- tuple of values (A_prev, W, b)
                    activation_cache -- the activation cache
        """ 
        Z, linear_cache = self.forward_linear(X)
        AL, activation_cache = self.forward_activation(Z)

        return AL, (linear_cache, activation_cache)
    


    def backward(self, AL, Y, caches, learning_rate):
        """Backward pass through softmax + fully connected layer"""

        m = AL.shape[1]  # Number of examples
        linear_cache, activation_cache = caches  # Retrieve caches

        X, W, b = linear_cache  # Extract stored values
        Z = activation_cache  # Z is not used for softmax 

        dAL = AL - Y  # Gradient of loss w.r.t softmax output
        dW = np.dot(dAL, X.T) / m  # Gradient of weights 
        db = np.sum(dAL, axis=1, keepdims=True) / m  # Gradient of biases
        dX = np.dot(W.T, dAL)  # Gradient for previous layer

        # Update parameters
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dX


    def compute_cost(self, AL, Y):
        '''Compute the cross-entropy cost'''
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL + 1e-9)) / m
        return cost


