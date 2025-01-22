import matplotlib.pyplot as plt
import torch

import sys
sys.path.append(r'C:\Users\matia\OneDrive\Documentos\GitHub\Memory-based-numerical-integration\Pytorch Implementation')
from Neural_Network import Neural_Network
from datetime import datetime
from numpy import pi, exp 

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)

## ----------------------Neural Network Parameters------------------##

deep_layers = 4
hidden_layers_dimension = 10

## ----------------------Training Parameters------------------##

batch_size = 3000
epochs = 2000
learning_rate = 10e-3
optimizer = "Adam"  # Adam or SGD

NN = Neural_Network(input_dimension = 1,
                    output_dimension = 1,
                    deep_layers = deep_layers,
                    hidden_layers_dimension = hidden_layers_dimension,
                    optimizer = "Adam",
                    learning_rate = learning_rate)

NN_memory = Neural_Network(input_dimension = 1,
                           output_dimension = 1,
                           deep_layers = deep_layers,
                           hidden_layers_dimension = hidden_layers_dimension,
                           optimizer = "Adam",
                           learning_rate = learning_rate) 
params = NN.state_dict()

NN_memory.load_state_dict(params)

rhs = lambda x: pi**2* torch.sin(pi * x * 2)

beta = lambda t: exp(-0.001 * t) + 0.001

last_loss = 0.
last_loss_list = []
beta_list = []

def compute_loss(nb_iter, last_loss):
    
    x = torch.rand(batch_size)
        
    integrad_value = (NN.laplacian(x) + rhs(x))**2
    
    integral_value_memory = (NN_memory.laplacian(x) + rhs(x))**2
    loss_value_memory = beta(nb_iter) * integral_value_memory - (beta(nb_iter) - 1) * last_loss # cambie + (beta(nb_iter) - 1) a - (beta(nb_iter) - 1)
    
    boundary_value = NN(torch.tensor([0., 1.]))**2 
    
    boundary_value_memory = NN_memory(torch.tensor([0., 1.]))**2 
        
    loss = torch.nn.L1Loss(reduction = 'mean')
    
    res = torch.concat([integrad_value, boundary_value], axis = -1)
    
    res_memory = torch.concat([loss_value_memory, boundary_value_memory], axis = -1)
        
    res_value = loss(res,torch.zeros_like(res,requires_grad = True))
            
    res_value_memory = loss(res_memory,torch.zeros_like(res_memory,requires_grad = True))
    
    return res_value, res_value_memory


loss_list = []
loss_memory_list = []

start_time = datetime.now()

print(f"{'='*30} Training {'='*30}")
for epoch in range(epochs):
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"{'='*20} [{current_time}] Epoch:{epoch + 1}/{epochs} {'='*20}")
    
    loss_value, loss_value_memory =  compute_loss(epoch, last_loss)
    
    last_loss = loss_value_memory.item()
    
    loss_list.append(loss_value.item())
    loss_memory_list.append(last_loss)
    beta_list.append(beta(epoch))
    
    print(f"Loss: {loss_value.item():.8f} Loss with memory: {loss_value_memory.item():.8f}")
    
    NN.optimizer_step(loss_value)
    
    NN_memory.optimizer_step(loss_value_memory)
    
    
end_time = datetime.now()

execution_time = end_time - start_time

print(f"Training time: {execution_time}")

plot_points = torch.linspace(0, 1, 1000)

figure_solution, axis_solution = plt.subplots(dpi=500)

exact = lambda x: torch.sin(pi * x * 2) / 4

exact_evaluation = exact(plot_points)
exact_evaluation = exact_evaluation.cpu().detach().numpy()

NN_evaluation = NN(plot_points)
NN_evaluation = NN_evaluation.cpu().detach().numpy()

NN_memory_evaluation = NN_memory(plot_points)
NN_memory_evaluation = NN_memory_evaluation.cpu().detach().numpy()

plot_points = plot_points.cpu().detach().numpy()

axis_solution.plot(plot_points,
                   exact_evaluation,
                   alpha = 0.6,
                   label = "exact")

axis_solution.plot(plot_points,
                   NN_evaluation,
                   linestyle = "-.",
                   label = r"$u^{\theta}$")

axis_solution.plot(plot_points,
                   NN_memory_evaluation,
                   linestyle = ":",
                   label = r"$u_{M}^{\theta}$")

axis_solution.legend()

axis_solution.set(title = "PINNs final solution", xlabel = "x", ylabel = "u(x)")

figure_error, axis_error = plt.subplots(dpi=500)

axis_error.semilogy(loss_list, 
                    linestyle = "-.", 
                    label = r"$\mathcal{L}^{\theta}$")

axis_error.semilogy(loss_memory_list, 
                    linestyle = ":", 
                    label = r"$\mathcal{L}_{M}^{\theta}$")

axis_error.legend()

axis_error.set(title = "Loss Evolution", xlabel = "# Epoch", ylabel = "loss value")

figure_beta, axis_beta = plt.subplots(dpi = 500)

axis_beta.set(title = "Beta Evolution", xlabel = "# Epoch", ylabel = "Beta value")

axis_beta.semilogy(beta_list)

plt.show()