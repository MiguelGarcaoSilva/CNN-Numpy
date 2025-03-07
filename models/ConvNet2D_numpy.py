import numpy as np

class ConvNet2D:
    def __init__(self, in_channels, out_channels, num_filters, filter_sizes, strides, paddings, pooling = None,
                 pool_sizes=None, pooling_strides=None):
        """CNN classifier with parameterizable number of convolutional layers and pooling layers, and a fully connected layer
        
        Arguments:
        in_channels -- Number of channels in the input image
        out_channels -- Number of channels produced by the convolution
        num_filters -- List of number of filters in each convolutional layer
        filter_sizes -- List of filter sizes in each convolutional layer
        strides -- List of strides in each convolutional layer
        paddings -- List of paddings in each convolutional layer
        pooling -- Pooling type ('max' or 'average') or None
        pool_sizes -- List of pooling sizes in each pooling layer
        pooling_strides -- List of pooling strides in each pooling layer    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.strides = strides
        self.paddings = paddings
        self.pooling = pooling
        self.pool_sizes = pool_sizes
        self.pooling_strides = pooling_strides

        assert len(num_filters) == len(filter_sizes) == len(strides) == len(paddings)
        assert pooling is None or (pool_sizes is not None and len(pool_sizes) == len(num_filters))

        self.conv_layers = []
        self.pooling_layers = []

        for i in range(len(num_filters)):
            self.conv_layers.append(Conv2d(in_channels, num_filters[i], filter_sizes[i], strides[i], paddings[i]))
            in_channels = num_filters[i]  # Update in_channels for next layer

            if pooling is not None and pool_sizes[i]:
                self.pooling_layers.append(Pooling(pool_sizes[i], pooling_strides[i], pooling))

        self.fc = None # initialized in foward pass because we don't know the input size yet


    def forward(self, A):
        """Forward pass through the CNN
        Arguments:
        A -- Input data of shape (m, n_H, n_W, n_C)
        Returns:
        A -- Output of the fully connected layer, shape (n_y, m)
        caches -- List of caches for each layer
        """
        caches = []

        # Forward pass through convolutional + pooling layers
        for i in range(len(self.conv_layers)):
            A, (cache_conv, cache_activation) = self.conv_layers[i].forward(A)
            caches.append((cache_conv, cache_activation)) 

            if len(self.pooling_layers) > 0:
                A, cache_pool = self.pooling_layers[i].forward(A)
                caches.append(cache_pool)  

        # Flatten feature map for fully connected layer
        A = A.reshape(A.shape[0], -1).T  # (n_x, m)
        # Initialize fully connected layer only the first time
        if self.fc is None:
            self.fc = FullyConnected(A.shape[0], self.out_channels)
        A, (fc_linear_cache, fc_activation_cache) = self.fc.forward(A)
        caches.append((fc_linear_cache, fc_activation_cache))  

        return A, caches


    def backward(self, caches, AL, Y):
        """Backward pass through the CNN
        Arguments:
        caches -- List of caches for each layer
        AL -- Output of the fully connected layer, shape (n_y, m)
        Y -- True "label" vector, shape (n_y, m)
        Returns:
        Grads -- Dictionary of gradients for each layer
        """

        grads = {}
        m = AL.shape[1]

        # Backprop through fully connected layer
        fc_linear_cache, fc_activation_cache = caches.pop()
        Y = Y.reshape(AL.shape)  # Ensure correct shape
        dA, dW, db = self.fc.backward(AL, Y, (fc_linear_cache, fc_activation_cache))

        grads['dW'] = dW
        grads['db'] = db

        # Backprop through convolutional and pooling layers
        for i in reversed(range(len(self.conv_layers))):
            if len(self.pooling_layers) > 0:
                cache_pool = caches.pop()  
                dA = self.pooling_layers[i].backward(dA, cache_pool)

            cache_conv, cache_activation = caches.pop()  
            dA, dW, db = self.conv_layers[i].backward(dA, (cache_conv, cache_activation))

            grads[f'dW{i}'] = dW
            grads[f'db{i}'] = db

        return grads

    def update_parameters_adam(self, grads, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, t=0):
        """Update the weights using the Adam optimizer"""
        for i, conv in enumerate(self.conv_layers):
            conv.W, conv.W_m, conv.W_v = self.update_parameters_adam_step(conv.W, grads[f'dW{i}'], conv.W_m, conv.W_v, learning_rate, beta1, beta2, epsilon, t)
            conv.b, conv.b_m, conv.b_v = self.update_parameters_adam_step(conv.b, grads[f'db{i}'], conv.b_m, conv.b_v, learning_rate, beta1, beta2, epsilon, t)

        self.fc.W, self.fc.W_m, self.fc.W_v = self.update_parameters_adam_step(self.fc.W, grads['dW'], self.fc.W_m, self.fc.W_v, learning_rate, beta1, beta2, epsilon, t)
        self.fc.b, self.fc.b_m, self.fc.b_v = self.update_parameters_adam_step(self.fc.b, grads['db'], self.fc.b_m, self.fc.b_v, learning_rate, beta1, beta2, epsilon, t)

    @staticmethod
    def update_parameters_adam_step(theta, dtheta, m, v, learning_rate, beta1, beta2, epsilon, t):
        """Update the weights using the Adam optimizer for a single layer"""
        m = beta1 * m + (1 - beta1) * dtheta
        v = beta2 * v + (1 - beta2) * (dtheta ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        theta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        return theta, m, v

    def compute_cost(self, AL, Y):
        '''Compute the cross-entropy cost
        Arguments:
        AL -- Probability vector corresponding to the label predictions, shape (n_y, m)
        Y -- True "label" vector, shape (n_y, m)
        Returns:
        cost -- Cross-entropy cost
        '''
    
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL + 1e-9)) / m
        return cost

    def fit(self, X, Y, learning_rate=0.01, n_iters=1000, seed=0):
        '''Fit according to the learning rate and number of iterations
        
        Arguments:
        X -- Input data of shape (m, n_H, n_W, n_C)
        Y -- True "label" vector of shape (m, n_y)
        learning_rate -- Learning rate of the optimization
        n_iters -- Number of iterations of the optimization loop
        seed -- Random seed for reproducibility
        '''
        np.random.seed(seed)
        m = X.shape[0]
        input_dim = X.shape[3]
        output_dim = Y.shape[1] 
        costs = []

        assert Y.shape[1] == self.out_channels

        for i in range(n_iters):
            AL, cache = self.forward(X)
            cost = self.compute_cost(AL, Y.T)
            grads = self.backward(cache, AL, Y.T)
            self.update_parameters_adam(grads, learning_rate, t=i+1)
            costs.append(cost)
            if i % 10 == 0:
                print(f'Cost after iteration {i}: {cost}')

        return costs

    def predict(self, X):
        """Returns the predicted class index for each sample.
        Arguments:
        X -- Input data of shape (m, n_H, n_W, n_C)
        Returns:
        Predictions -- Array of predicted class indices
        """
        probs = self.forward(X)[0]
        return np.argmax(probs, axis=0)

class Conv2d:

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Applies a 2D convolution over an input signal composed of several input planes.

        Arguments:
        in_channels -- Number of channels in the input image
        out_channels -- Number of channels produced by the convolution
        kernel_size -- Size of the convolving kernel
        stride -- Stride of the convolution. Default: 1
        padding -- Zero-padding added to both sides of the input. Int or 'valid' or 'same'. Default: 0
        """

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.n_filters = out_channels
        self.hparameters = {"stride": stride, "pad": padding}
        # each filter fxf with n_C_prev channels will produce one output channel
        self.W = np.random.randn(kernel_size, kernel_size, in_channels, out_channels) * np.sqrt(2. / (kernel_size * kernel_size * in_channels))  # (f, f, n_C_prev, n_C) 
        # each filter will have one bias
        self.b = np.zeros((1, 1, 1, out_channels))  # (1, 1, 1, n_C)


        self.W_m, self.W_v = np.zeros_like(self.W), np.zeros_like(self.W)
        self.b_m, self.b_v = np.zeros_like(self.b), np.zeros_like(self.b)

    @staticmethod
    def ReLU(Z):
        """ReLU activation function"""
        return np.maximum(0, Z)


    @staticmethod
    def zero_pad(X, pad):
        """
        Pad with zeros all 2D data of the dataset X. The padding is applied to both dimensions

        Arguments:
        X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m 2D objects
        pad -- integer, amount of padding around each object on both dimensions

        Returns:
        X_pad -- padded object of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
        """
        return np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))

    @staticmethod 
    def conv_single_step(slice, W, b):
        """
        Apply one filter defined by parameters W on a single slice (slice) of the output activation of the previous layer.

        Arguments:
        slice -- slice of input data of shape (f, f, n_C_prev)
        W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
        b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

        Returns:
        Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
        """
        return np.sum(np.multiply(slice, W)) + b



    def forward(self, A_prev):
        """
        Implements the forward propagation for a convolution function
        
        Arguments:
        A_prev -- output activations of the previous layer (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
        b -- Biases, numpy array of shape (1, 1, 1, n_C)
        hparameters -- python dictionary containing "stride" and "pad"
            
        Returns:
        A -- conv output, numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward() function
        """
        
        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve dimensions from W's shape
        (f, f, n_C_prev, n_C) = self.W.shape

        stride = self.hparameters["stride"]
        pad = self.hparameters["pad"]

        # Compute padding based on the mode
        if pad == 'same':
            pad = max((stride - 1) * n_H_prev + (f - 1) // 2, 0)
        elif pad == 'valid':  # 'valid' mode (no padding)
            pad = 0
        else:
            pass # Assume padding is an integer

        # Padding the A_prev
        A_prev_pad = self.zero_pad(A_prev, pad)

        # Compute the dimensions of the CONV output volume using the formula. 
        n_H = int(np.floor((n_H_prev - f + 2 * pad) / stride) + 1)
        n_W = int(np.floor((n_W_prev - f + 2 * pad) / stride) + 1)

        # Initialize the output volume A (Z after activation) and Z with zeros
        Z = np.zeros((m, n_H, n_W, n_C))
        A = np.zeros((m, n_H, n_W, n_C))

        for i in range(m):               # loop training examples
            A_prev_pad_i = A_prev_pad[i]     
            for h in range(n_H):           # loop over vertical axis of the output volume
                #Find the vertical start and end of the current "slice"
                vert_start = h * stride
                vert_end = vert_start + f

                for w in range(n_W):       # loop over horizontal axis of the output volume
                    #Find the horizontal start and end of the current "slice"
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    for c in range(n_C):   # loop over channels (= #filters) of the output volume

                        #Get the 3d slice of the example i at the current position (h, w)
                        a_slice_prev = A_prev_pad_i[vert_start:vert_end, horiz_start:horiz_end, :]

                        #Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
                        weights = self.W[:,:,:,c]       #(f, f, n_C_prev)
                        biases = self.b[:,:,:,c]        #(1, 1, 1)
                        Z[i, h, w, c] = self.conv_single_step(a_slice_prev, weights, biases)
                        # Activation
                        A[i, h, w, c] = self.ReLU(Z[i, h, w, c])


        # Save information in "cache" for the backprop
        conv_cache = (A_prev, self.W.copy(), self.b.copy(), self.hparameters.copy())
        activation_cache = Z

        return A, (conv_cache, activation_cache)


    def backward(self, dA, cache):
        """
        Implement the backward propagation for a convolution function
        
        Arguments:
        dA -- gradient of the cost with respect to the output of the conv layer (A), numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward(), output of forward()
        
        Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        dW -- gradient of the cost with respect to the weights of the conv layer (W)
            numpy array of shape (f, f, n_C_prev, n_C)
        db -- gradient of the cost with respect to the biases of the conv layer (b)
            numpy array of shape (1, 1, 1, n_C)
        """    
        
        # Retrieve information from "cache"
        (conv_cache, activation_cache) = cache
        Z = activation_cache
        dA = dA.reshape(Z.shape)
        dZ = dA * (Z > 0)  # ReLU derivative: 1 if Z > 0, else 0

        A_prev, W, b, hparameters = conv_cache
        
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape         # Retrieve dimensions from A_prev's shape
        (f, f, n_C_prev, n_C) = W.shape         # Retrieve dimensions from W's shape
        
        stride = hparameters["stride"]
        pad = hparameters["pad"]

        # Compute padding based on the mode
        if pad == 'same':
            pad = max((stride - 1) * n_H_prev + (f - 1) // 2, 0)
        elif pad == 'valid':  # 'valid' mode (no padding)
            pad = 0
        else:
            pass # Assume padding is an integer
        
        (m, n_H, n_W, n_C) = dA.shape  # Retrieve dimensions from dA's shape

        # Initialize dA_prev, dW, db 
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                         
        dW = np.zeros((f, f, n_C_prev, n_C)) 
        db = np.zeros((1, 1, 1, n_C)) 
        
        # Pad A_prev and dA_prev
        A_prev_pad = self.zero_pad(A_prev, pad)
        dA_prev_pad = self.zero_pad(dA_prev, pad)
        
        for i in range(m):  # loop over the training examples
            
            #ith example from A_prev_pad and dA_prev_pad
            a_prev_pad_i = A_prev_pad[i]
            da_prev_pad_i = dA_prev_pad[i]
            
            for h in range(n_H):                   # loop over vertical axis of the output volume
                vert_start = h * stride
                vert_end = h * stride + f
                for w in range(n_W):               # loop over horizontal axis of the output volume
                    horiz_start = w * stride
                    horiz_end = w * stride + f
                    for c in range(n_C):           # loop over the channels of the output volume
                        
                        #define the slice from a_prev_pad
                        a_slice = a_prev_pad_i[vert_start:vert_end, horiz_start:horiz_end, :]

                        #Update gradients for the window and the filter's parameters

                        #formula for computing 𝑑𝐴 with respect to the cost for a certain filter 𝑊𝑐 and a given training example
                        da_prev_pad_i[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c] 
                        dW[:,:,:,c] += a_slice * dZ[i, h, w, c]  #formula for computing dw (derivative of one filter 𝑊𝑐 with respect to loss)
                        db[:,:,:,c] += dZ[i, h, w, c] #formula for computing db (derivative of one filter 𝑊𝑐 with respect to loss)
                        
            #Set the ith training example's dA_prev to the unpadded da_prev_pad
            if pad == 0:
                dA_prev[i, :, :, :] = da_prev_pad_i
            else:  
                dA_prev[i, :, :, :] = da_prev_pad_i[pad:-pad, pad:-pad, :]
        
        return dA_prev, dW, db


class Pooling:

    def __init__(self, f, stride, mode = 'max'):

        self.hparameters = {"f":f ,"stride": stride}
        self.mode = mode

        assert self.mode in ['max', 'average']
    
    def forward(self, A_prev):
        """
        Implements the forward pass of the pooling layer
        
        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        hparameters -- python dictionary containing "f" and "stride"
        mode -- the pooling mode defined as a string ("max" or "average")
        
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
                for w in range(n_W):                 # loop on the horizontal axis
                    #Find the horizontal start and end of the current "slice"
                    horiz_start = w*stride
                    horiz_end = w*stride + f
                    for c in range (n_C):            # loop over the channels 
                        # Get current slide on the ith training example of A_prev, channel c
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]  
                        
                        #Compute the pooling operation on the slice. 
                        if self.mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif self.mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)
        
        
        # Store the input and hparameters in "cache" for pool_backward()
        cache = (A_prev, self.hparameters.copy())
        
        return A, cache

    def backward(self, dA, cache):
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
                vert_start = h*stride
                vert_end = h*stride + f
                for w in range(n_W):               # loop on the horizontal axis
                    horiz_start = w*stride
                    horiz_end = w*stride + f
                    for c in range(n_C):           # loop over the channels (depth)

                        #Compute the backward propagation in both modes.
                        if self.mode == "max":
                            
                            #Use the corners and "c" to define the current slice from a_prev
                            a_prev_slice =  a_prev_i[vert_start:vert_end, horiz_start:horiz_end, c] 

                            # Only the maximum value in each pooling window influences the output.
                            # We use a mask to determine which element in the input slice was selected as max.
                            # We assign the gradient only to that max value in dA_prev.
                                                        
                            mask = (a_prev_slice == np.max(a_prev_slice))
                            mask /= np.sum(mask)  # Normalize the mask when multiple elements are equal

                            #Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += dA[i, h, w, c] * mask
                            
                        elif self.mode == "average":
                            
                            #Get the value da from dA (≈1 line)
                            da = dA[i, h, w, c]

                            #Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da.

                            #Each element in the input pooling window equally influences the output.
                            #The gradient is evenly distributed across all elements in the pooling region.
                            average = da / (f * f)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += average # broadcasted to fxf
        
        return dA_prev
    

class FullyConnected:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim #(n_x)
        self.output_dim = output_dim #(n_y)
        self.W = np.random.randn(self.output_dim, self.input_dim) * np.sqrt(1. / self.input_dim)  # shape: (n_y, n_x)
        self.b = np.zeros((self.output_dim, 1)) # shape: (n_y, 1)

        self.W_m, self.W_v = np.zeros_like(self.W), np.zeros_like(self.W)
        self.b_m, self.b_v = np.zeros_like(self.b), np.zeros_like(self.b)
    @staticmethod
    def softmax(x):
        '''Softmax activation function with cache'''
        exps = np.exp(x - np.max(x)) #subtracting max(x) to avoid numerical instability
        return exps / np.sum(exps, axis=0)

    def forward_linear(self, A):
        '''Forward pass through the
        linear layer'''
        Z = np.dot(self.W, A) + self.b  # shape: (n_y, m) = (n_y, n_x) * (n_x, m) + (n_y, 1)

        W , b = self.W.copy(), self.b.copy()
        linear_cache = (A, W, b)
        return Z, linear_cache

    def forward_activation(self, Z):
        '''Forward pass through the
        activation function'''
        A = self.softmax(Z) # shape: (n_y, m)
        return A, Z


    def forward(self, A_prev):
        """
        Implement forward propagation for the LINEAR->SOFTMAX computation
        
        Arguments:
        A_prev -- activations from previous layer (or input data): (n_x, m)
        
        Returns:
        AL -- activation value from the output (last) layer (n_y, m)
        caches -- list of caches containing:
                    linear_cache -- tuple of values (A_prev, W, b)
                    activation_cache -- the activation cache
        """ 
        Z, linear_cache = self.forward_linear(A_prev)
        AL, activation_cache = self.forward_activation(Z)

        return AL, (linear_cache, activation_cache)
    


    def backward(self, AL, Y, caches):
        """Backward pass through softmax + fully connected layer
        Arguments:
        AL -- Probability vector, output of the forward propagation (L_model_forward()) shape: (n_y, m)
        Y -- True "label" vector  shape: (n_y, m)
        caches -- list of caches containing:
                    linear_cache -- tuple of values (A_prev, W, b)
                    activation_cache -- the activation cache

        Returns:
        dA_prev -- Gradient of the loss with respect to the input of the fully connected layer
        """

        m = AL.shape[1]  # Number of examples
        (linear_cache, activation_cache) = caches  # Retrieve caches

        X, W, b = linear_cache  # Extract stored values
        Z = activation_cache  # Z is not used for softmax 

        dAL = AL - Y  # Gradient of loss w.r.t softmax output
        dW = np.dot(dAL, X.T) / m  # Gradient of weights 
        db = np.sum(dAL, axis=1, keepdims=True) / m  # Gradient of biases
        dA_prev = np.dot(W.T, dAL)  # Gradient for previous layer

        return dA_prev, dW, db



