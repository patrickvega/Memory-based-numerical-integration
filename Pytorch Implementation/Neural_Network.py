import torch

torch.set_default_dtype(torch.float64)

class Neural_Network(torch.nn.Module):
    """
    Neural Network class for constructing a deep neural network with configurable layers, optimizer, and learning rate schedule.

    Parameters:
    - input_dimension (int): Input dimension of the network (default is 2).
    - output_dimension (int): Output dimension of the network (default is 1).
    - deep_layers (int): Number of hidden layers (default is 5).
    - hidden_layers_dimension (int): Dimension of each hidden layer (default is 25).
    - activation_function (torch.nn.Module): Activation function (default is torch.nn.Tanh()).
    - optimizer (str): Optimizer name (default is "Adam").
    - learning_rate (float): Learning rate (default is 0.0005).
    - scheduler (str): Type of learning rate scheduler (default is "None").
    - decay_rate (float): Decay rate for learning rate scheduler (default is 0.9).
    - decay_steps (int): Number of steps for learning rate decay (default is 200).
    """
    def __init__(self, 
                 input_dimension: int,
                 output_dimension: int,         
                 deep_layers: int = 5,      
                 hidden_layers_dimension: int = 25,      
                 activation_function: torch.nn.Module = torch.nn.Tanh(),  
                 optimizer: str = "Adam",   
                 learning_rate: float = 0.0005,
                 scheduler: str = "None",
                 decay_rate: float = 0.95,
                 decay_steps: int = 100,
                 use_xavier : bool = False
                 ):
        
        super().__init__()  
        
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.layer_in = torch.nn.Linear(input_dimension, 
                                        hidden_layers_dimension)
        self.layer_out = torch.nn.Linear(hidden_layers_dimension, 
                                         output_dimension)
        self.middle_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_layers_dimension, 
                                                                  hidden_layers_dimension) for _ in range(deep_layers)])
        self.activation_function = activation_function
                        
        self.optimizer_name = optimizer  
        self.learning_rate = learning_rate  
        
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), 
                                              lr=self.learning_rate)  
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), 
                                             lr=self.learning_rate)  

        if scheduler == "Exponential":
            self.gamma = decay_rate **(1/decay_steps)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 
                                                                    self.gamma)
        else:
            self.scheduler = None
            
        if use_xavier == True:
            self._initialize_weights()
            
    def forward(self, 
                *inputs: torch.Tensor):
        """
        Forward pass with flexible number of input tensors (1 to 3).
        
        Parameters:
        - inputs (torch.Tensor): Variable number of input tensors.
        
        Returns:
        - torch.Tensor: Output tensor after passing through the network.
        """
        if len(inputs) == 1:
            Input = inputs[0].unsqueeze(-1)
        else:
            Input = torch.stack(inputs, dim = -1)
        
        output = self.activation_function(self.layer_in(Input))
        
        for layer in self.middle_layers:
            output = self.activation_function(layer(output))
        
        return self.layer_out(output).squeeze(-1)
                    
    def optimizer_step(self, 
                       loss_value: torch.Tensor):
        """
        Perform an optimization step.
        
        Parameters:
        - loss_value (torch.Tensor): The loss value to minimize.
        
        Updates:
        - Model parameters using the optimizer.
        """
        if self.optimizer_name == "LBFGS":
            def closure():
                self.optimizer.zero_grad()
                loss = loss_value
                loss.backward(retain_graph=True)
                return loss
            self.optimizer.step(closure)
        else:
            self.optimizer.zero_grad()
            loss_value.backward(retain_graph=True)
            self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
            
    def update_optimizer(self, optimizer_name: str, learning_rate: float = None):
       self.optimizer_name = optimizer_name
       if learning_rate:
           self.learning_rate = learning_rate
           
       if optimizer_name == "Adam":
           self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
       elif optimizer_name == "SGD":
           self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
       elif optimizer_name == "LBFGS":
           self.optimizer = torch.optim.LBFGS(self.parameters(), lr=self.learning_rate)
       else:
           raise ValueError(f"Optimizer {optimizer_name} not recognized.")
    
    def grad(self, 
             *inputs: torch.Tensor):
        """
        Compute grad w.r.t inputs.
        
        Parameters:
        - inputs (torch.Tensor)
        
        Returns:
        - Tuple of gradients w.r.t each input tensor (torch.Tensor)
        """
        for input_tensor in inputs:
            input_tensor.requires_grad_(True)
        
        output = self.forward(*inputs)
        
        gradients = torch.autograd.grad(outputs = output,
                                        inputs = inputs,
                                        grad_outputs = torch.ones_like(output),
                                        retain_graph  = True,
                                        create_graph = True)
        
        return gradients[0]
                
    def laplacian(self, *inputs: torch.Tensor):
        """
        Compute Laplacian of Neural Network w.r.t inputs.
        
        Parameters:
        - inputs (torch.Tensor): input tensors.
        
        Returns:
        - laplacian (torch.Tensor): The Laplacian of the network output w.r.t each input tensor.
        """
        for input_tensor in inputs:
            input_tensor.requires_grad_(True)
        
        output = self.forward(*inputs)
        
        gradients = torch.autograd.grad(outputs = output,
                                        inputs = inputs,
                                        grad_outputs = torch.ones_like(output),
                                        retain_graph = True,
                                        create_graph = True)
        
        second_derivatives = []
        
        for grad, input_tensor in zip(gradients, inputs):
            grad2 = torch.autograd.grad(outputs = grad,
                                        inputs = input_tensor,
                                        grad_outputs = torch.ones_like(grad),
                                        retain_graph = True,
                                        create_graph = True)[0]
            second_derivatives.append(grad2)
        
        laplacian = sum(second_derivatives)
        
        return laplacian
        
    def partial(self, input_idx: int, *inputs: torch.Tensor):
        """
        Compute partial derivatives of each output of the Neural Network w.r.t a specific input.
        
        Parameters:
        - input_idx (int): Index of the input with respect to which the derivative will be computed (0 or 1).
        - inputs (torch.Tensor): Variable number of input tensors.
        
        Returns:
        - partial_derivs (torch.Tensor): Tensor of partial derivatives for each output w.r.t the selected input.
        """
        for i, input_tensor in enumerate(inputs):
            if i == input_idx:
                input_tensor.requires_grad_(True)
        
        output = self.forward(*inputs)
        
        if self.output_dimension == 1:
            partial_deriv = torch.autograd.grad(outputs = output, 
                                                inputs = inputs[input_idx],
                                                grad_outputs = torch.ones_like(output), 
                                                create_graph = True)[0]
            
            return partial_deriv
        
        else:
        
            partial_derivs = []
            
            for i in range(output.shape[-1]): 
                partial_deriv = torch.autograd.grad(outputs=output[..., i], 
                                                    inputs=inputs[input_idx],  
                                                    grad_outputs=torch.ones_like(output[..., i]), 
                                                    create_graph=True)[0]
                partial_derivs.append(partial_deriv)
            
            return torch.stack(partial_derivs, dim=-1)
        
    def second_partial(self, input_idx: int, *inputs: torch.Tensor):
        """
        Compute second partial derivatives of each output of the Neural Network w.r.t a specific input.
        
        Parameters:
        - input_idx (int): Index of the input with respect to which the second derivative will be computed (0 o 1).
        - inputs (torch.Tensor): Variable number of input tensors.
        
        Returns:
        - second_partial_derivs (torch.Tensor): Tensor of second partial derivatives for each output w.r.t the selected input.
        """
        for i, input_tensor in enumerate(inputs):
            if i == input_idx:
                input_tensor.requires_grad_(True)
        
        output = self.forward(*inputs)
        
        if self.output_dimension == 1:
            partial_deriv = torch.autograd.grad(outputs = output,
                                                inputs = inputs[input_idx],
                                                grad_outputs = torch.ones_like(output),
                                                create_graph = True)[0]
            
            second_partial_deriv = torch.autograd.grad(outputs = partial_deriv,
                                                       inputs = inputs[input_idx],
                                                       grad_outputs = torch.ones_like(partial_deriv),
                                                       create_graph = True)[0]
            
            return second_partial_deriv
        
        else:
            second_partial_derivs = []
            
            for i in range(output.shape[-1]):
                partial_deriv = torch.autograd.grad(outputs=output[..., i], 
                                                    inputs=inputs[input_idx], 
                                                    grad_outputs=torch.ones_like(output[..., i]), 
                                                    create_graph=True)[0]
                
                second_partial_deriv = torch.autograd.grad(outputs = partial_deriv,
                                                           inputs=inputs[input_idx],
                                                           grad_outputs=torch.ones_like(partial_deriv),
                                                           create_graph=True)[0]
                
                second_partial_derivs.append(second_partial_deriv)
            
            return torch.stack(second_partial_derivs, dim=-1)

    # def update_optimizer(self, 
    #                      optimizer_name: str ,
    #                      learning_rate: float = None):
        
    #     if optimizer_name == "L-BFGS":
    #         self.optimizer = torch.optim.LBFGS(self.parameters(), 
    #                                            lr = self.learning_rate)  
    
    def _initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.layer_in.weight)
        torch.nn.init.xavier_uniform_(self.layer_out.weight)
        for layer in self.middle_layers:
            torch.nn.init.xavier_uniform_(layer.weight)